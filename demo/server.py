"""Demo website backend.

Upload a patient record -> a digital twin is built from the 5 most similar training patients ->
the twin predicts how effective every medication would be, and the RL agent picks one.

Run from the project folder:  python -m demo   (then open http://localhost:8000)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from digital_twin_rl.__main__ import AGENT, HISTORIES, MODEL, REGIMENS
from digital_twin_rl.data import load_jsonl
from digital_twin_rl.model import OutcomeModel
from digital_twin_rl.record import RecordError, parse_record
from digital_twin_rl.rl import allowed, best_action, load_agent, observe, platinum_resistant
from digital_twin_rl.twin import NeighbourIndex, build_twin

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"   # the built website
MAX_UPLOAD_BYTES = 200_000

# ---------------------------------------------------------------------------------------------
# Display names shown on the website.
# ---------------------------------------------------------------------------------------------
# The twin works on drug classes; the drug names are typical examples of each class.
MEDICATIONS = {
    "platinum+taxanes": ("Carboplatin + Paclitaxel", "Platinum + taxane"),
    "anti_angiogenics+platinum+taxanes": ("Carboplatin + Paclitaxel + Bevacizumab", "Platinum + taxane + anti-angiogenic"),
    "antimetabolites+platinum+taxanes": ("Carboplatin + Paclitaxel + Gemcitabine", "Platinum + taxane + antimetabolite"),
    "platinum": ("Carboplatin alone", "Platinum"),
    "antimetabolites+platinum": ("Carboplatin + Gemcitabine", "Platinum + antimetabolite"),
    "alkylating_agents+platinum": ("Carboplatin + Cyclophosphamide", "Platinum + alkylating agent"),
    "taxanes": ("Paclitaxel alone", "Taxane"),
    "anthracyclines": ("Liposomal Doxorubicin", "Anthracycline"),
    "topoisomerase_inhibitors": ("Topotecan", "Topoisomerase inhibitor"),
    "topoisomerase_inhibitors+vinca_alkaloids": ("Etoposide + Vinorelbine", "Topoisomerase inhibitor + vinca alkaloid"),
    "antimetabolites": ("Gemcitabine", "Antimetabolite"),
    "alkylating_agents": ("Cyclophosphamide", "Alkylating agent"),
    "hormonal_agents": ("Tamoxifen or Letrozole", "Hormonal therapy"),
    "radiation": ("Radiotherapy", "Radiation"),
}
SITUATIONS = {"primary": "Newly diagnosed", "recurrent": "Recurrence", "progressive": "Progression",
              "unknown": "Not recorded"}
RESIDUAL = {0: "None", 1: "1-10 mm", 2: "11-20 mm", 3: "> 20 mm"}


def class_names(classes) -> list[str]:
    # "anti_angiogenics" -> "Anti angiogenics"
    return [c.replace("_", " ").capitalize() for c in classes]


def age_years(state: dict) -> int | None:
    return None if state["age_years"] is None else round(state["age_years"])


# ---------------------------------------------------------------------------------------------
# Everything the website needs, loaded once when the server starts.
# ---------------------------------------------------------------------------------------------
class Lab:
    """The trained outcome model, the searchable training patients and the RL agent, loaded once."""

    def __init__(self):
        self.model = OutcomeModel.load(MODEL)
        self.index = NeighbourIndex(load_jsonl(HISTORIES), self.model)
        self.regimens = json.loads(REGIMENS.read_text(encoding="utf-8"))
        self.agent = load_agent(AGENT)

    def analyse(self, record: dict) -> dict:
        # 1. Read the uploaded record and build the patient's twin from the 5 most similar patients.
        parsed = parse_record(record)
        state = parsed["state"]
        twin = build_twin(state, self.model, self.index)

        # 2. Ask the twin about every medication; mark the ones guidelines advise against.
        mask = allowed(state, self.regimens)
        medications = []
        for regimen, ok in zip(self.regimens, mask):
            key = "+".join(regimen)
            name, group = MEDICATIONS.get(key, (key, ""))
            medications.append({"key": key, "name": name, "group": group, "guideline_caution": not ok,
                                **twin.predict(regimen)})
        # Most effective first (expected months alive within 5 years).
        medications.sort(key=lambda m: -m["survival_months"])

        # 3. Send back the patient summary, the 5 similar patients (their profile only, no ids or
        #    outcomes), the risk multipliers, the ranked medications and the RL agent's choice.
        platinum = state["months_since_last_platinum"]
        return {
            "label": parsed["label"], "today_month": parsed["today_month"],
            "patient": {
                "age": age_years(state), "stage": state["stage"], "grade": state["grade"],
                "residual": RESIDUAL.get(state["residual"], "Not recorded"),
                "treatment_line": state["line_number"], "situation": SITUATIONS[state["setting"]],
                "months_since_platinum": None if platinum is None else round(platinum, 1),
                "platinum_resistant": platinum_resistant(state),
                "past_treatments": [{"start_month": line["start_month"], "drugs": class_names(line["classes"])}
                                    for line in parsed["past_lines"]],
            },
            "neighbours": [{"similarity": round(n["similarity"], 3), "age": age_years(n["state"]),
                            "stage": n["state"]["stage"], "treatment_line": n["state"]["line_number"],
                            "situation": SITUATIONS[n["state"]["setting"]],
                            "past_drugs": class_names(sorted(n["state"]["prior_exposure"]))} for n in twin.neighbours],
            "risk_multipliers": {"progression": round(twin.multipliers["pfs"], 3),
                                 "death": round(twin.multipliers["os"], 3)},
            "medications": medications,
            "agent_choice": "+".join(self.regimens[best_action(self.agent, observe(state), mask)]),
        }


# ---------------------------------------------------------------------------------------------
# The web server: one API endpoint, plus the built website.
# ---------------------------------------------------------------------------------------------
def create_app(lab: Lab | None = None) -> FastAPI:
    app = FastAPI(title="Ovarian cancer digital twin demo")
    lab = lab or Lab()

    @app.post("/api/analyse")
    async def analyse(file: UploadFile = File(...)):
        # Read the uploaded file (with a size limit), parse it as JSON and analyse it.
        # Problems come back as HTTP 413/422 with a message the website shows to the user.
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large for a patient record")
        try:
            return lab.analyse(json.loads(content.decode("utf-8-sig")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HTTPException(422, f"Not a valid JSON file: {error}") from error
        except RecordError as error:
            raise HTTPException(422, str(error)) from error

    # Serve the built website at "/"; if it has not been built, say how to build it.
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/")
        def no_website():
            return JSONResponse({"message": "Website not built: run `pnpm install` and `pnpm build` in demo/web"})
    return app
