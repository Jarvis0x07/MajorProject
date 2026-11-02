"""Backend for the mid-semester demo (Objectives 1-3).

Upload a patient record -> a new digital twin is built on the spot from the 5
nearest stored twins -> choose a medication -> the twin predicts response,
progression, survival and side effects; it can also simulate the next five
years and show what the basic RL agent would choose.

Run from the repository root:  python -m demo   (then open http://localhost:8000)
"""
from __future__ import annotations

import json
import math
import random
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from digital_twin_rl.__main__ import DEFAULT_OUT, load_context
from digital_twin_rl.dqn import DQNPolicy, load_agent
from digital_twin_rl.environment import build_observation, safety_mask
from digital_twin_rl.neighbor_twin import K_NEIGHBORS, TwinDatabase, create_twin
from digital_twin_rl.outcome_model import HORIZON_MONTHS
from digital_twin_rl.policies import GuidelinePolicy
from digital_twin_rl.record import RecordError, parse_record
from digital_twin_rl.twin import PatientDigitalTwin

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
CREATED_TWINS_LOG = DEFAULT_OUT / "twins" / "demo_created_twins.jsonl"
MAX_TWINS_IN_MEMORY = 200
MAX_UPLOAD_BYTES = 200_000

# Display names for the 14 regimens; the twin works on drug classes, so drug names are examples.
MEDICATIONS = {
    "platinum+taxanes": ("Carboplatin + Paclitaxel", "Platinum + taxane doublet"),
    "anti_angiogenics+platinum+taxanes": ("Carboplatin + Paclitaxel + Bevacizumab", "Platinum doublet + anti-angiogenic"),
    "antimetabolites+platinum+taxanes": ("Carboplatin + Paclitaxel + Gemcitabine", "Platinum + taxane + antimetabolite"),
    "platinum": ("Carboplatin alone", "Single-agent platinum"),
    "antimetabolites+platinum": ("Carboplatin + Gemcitabine", "Platinum + antimetabolite"),
    "alkylating_agents+platinum": ("Carboplatin + Cyclophosphamide", "Platinum + alkylating agent"),
    "taxanes": ("Paclitaxel alone", "Single-agent taxane"),
    "anthracyclines": ("Liposomal Doxorubicin", "Anthracycline"),
    "topoisomerase_inhibitors": ("Topotecan", "Topoisomerase inhibitor"),
    "topoisomerase_inhibitors+vinca_alkaloids": ("Etoposide + Vinorelbine", "Topoisomerase inhibitor + vinca alkaloid"),
    "antimetabolites": ("Gemcitabine", "Antimetabolite"),
    "alkylating_agents": ("Cyclophosphamide", "Alkylating agent"),
    "hormonal_agents": ("Tamoxifen or Letrozole", "Hormonal therapy"),
    "radiation": ("Radiotherapy", "Radiation"),
}
CLASS_NAMES = {
    "platinum": "Platinum", "taxanes": "Taxane", "anthracyclines": "Anthracycline", "antimetabolites": "Antimetabolite",
    "topoisomerase_inhibitors": "Topoisomerase inhibitor", "alkylating_agents": "Alkylating agent",
    "anti_angiogenics": "Anti-angiogenic", "hormonal_agents": "Hormonal", "vinca_alkaloids": "Vinca alkaloid",
    "radiation": "Radiation", "other": "Other",
}
SETTING_NAMES = {"primary": "Newly diagnosed", "recurrent": "Recurrence", "progressive": "Progression",
                 "unknown": "Situation not recorded"}
RESIDUAL_NAMES = {0: "none", 1: "1-10 mm", 2: "11-20 mm", 3: "> 20 mm"}


def guideline_flags(state: dict, regimen: list[str]) -> list[str]:
    """Treatment constraints from standard ovarian-cancer practice (MDP constraint layer, Objective 3).

    The twin's predictions are associations learned from observational data;
    platinum was mostly given to patients with a better outlook, so the twin can
    favour platinum even where guidelines advise against it. These flags are
    shown next to the prediction and the AI doctor may not choose a flagged option.
    """
    flags = []
    pfi = state.get("months_since_last_platinum")
    if "platinum" in regimen and pfi is not None and pfi < 6 and state.get("setting") != "primary":
        flags.append("Guidelines advise against platinum when the cancer returned within 6 months of the last "
                     "platinum (platinum-resistant disease).")
    return flags


class MedicationRequest(BaseModel):
    medication: str


class SimulationRequest(BaseModel):
    medication: str
    runs: int = Field(200, ge=20, le=1000)


class DemoState:
    """Models and stored twins, loaded once at startup."""

    def __init__(self, out: Path = DEFAULT_OUT):
        context = load_context(out)
        self.out = out
        self.ensemble, self.toxicity = context["ensemble"], context["toxicity"]
        self.regimens: list[list[str]] = context["regimens"]
        self.keys = ["+".join(r) for r in self.regimens]
        missing = [k for k in self.keys if k not in MEDICATIONS]
        if missing:
            raise RuntimeError(f"No display name for regimens {missing}")
        self.database = TwinDatabase.load(out / "twins" / "twin_database.json")
        self.map_points, self.project = self.database.projection()
        network, _ = load_agent(out / "agents" / "dqn.pt")
        self.agent = DQNPolicy(network, "dqn")
        self.guideline = GuidelinePolicy(self.regimens)
        self.twins: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.info = self._project_info()

    def _project_info(self) -> dict:
        report = json.loads((self.out / "outcome_model_report.json").read_text(encoding="utf-8"))["test"]
        evaluation = json.loads((self.out / "evaluation" / "policy_evaluation.json").read_text(encoding="utf-8"))
        neighbor = json.loads((self.out / "twins" / "neighbor_twin_report.json").read_text(encoding="utf-8"))
        history = json.loads((self.out / "agents" / "training_history.json").read_text(encoding="utf-8"))
        cohort = json.loads((self.out / "cohort_summary.json").read_text(encoding="utf-8"))
        fidelity = evaluation["simulation_fidelity"]["by_split"]["train"]
        first_line = evaluation["results"]["first_line_evaluation_twin"]
        chosen = next(v for k, v in neighbor["validation"]["results"].items() if k.startswith("chosen"))
        shared = neighbor["validation"]["results"]["shared model only (no personalisation)"]
        return {
            "cohort": {"patients": cohort["patients"], "patients_with_lines": cohort["patients_with_lines"],
                       "treatment_lines": cohort["lines"], "stored_twins": len(self.database.entries),
                       "neighbors_per_twin": K_NEIGHBORS},
            "twin_accuracy": {
                "response_auroc": report["response"]["auroc_twin"],
                "survival_c_index": report["overall_survival"]["c_index_12m_risk"],
                "progression_c_index": report["progression_free_survival"]["c_index_12m_risk"],
                "mortality_12m_predicted": report["overall_survival"]["predicted_event_rate_12m"],
                "mortality_12m_observed": report["overall_survival"]["observed_event_rate_12m_known_status"],
            },
            "neighbor_twins": {"validation_progression_brier_shared": shared["pfs12"]["brier"],
                               "validation_progression_brier_neighbor": chosen["pfs12"]["brier"]},
            "simulation_fidelity": {"months": [int(k[:-1]) for k in fidelity["simulated"]],
                                    "simulated": list(fidelity["simulated"].values()),
                                    "observed": list(fidelity["observed"].values()),
                                    "patients": fidelity["patients"]},
            "rl": {
                "learning_curve": [{"episode": h["episode"], "mean_return": h["mean_return"]} for h in history["dqn"]],
                "first_line_survival_months": {name: first_line[name]["metrics"]["survival_months_5y"]["mean"]
                                               for name in ("random", "clinician", "guideline", "dqn")},
                "test_patients": first_line["dqn"]["patients"],
            },
            "medications": [{"key": k, "name": MEDICATIONS[k][0], "group": MEDICATIONS[k][1]} for k in self.keys],
            "twin_map": [[round(float(x), 4), round(float(y), 4)] for x, y in self.map_points],
        }

    # --- twins -------------------------------------------------------------------------------
    def create(self, record: dict) -> dict:
        parsed = parse_record(record)
        built = create_twin(parsed["patient"], self.ensemble, self.toxicity, self.database,
                            current_state=parsed["state"])
        twin_id = uuid.uuid4().hex[:12]
        stored = {"id": twin_id, "label": parsed["label"], "today_month": parsed["today_month"],
                  "patient": parsed["patient"], "state": parsed["state"], "offsets": built["offsets"],
                  "neighbors": built["neighbors"]}
        with self.lock:
            self.twins[twin_id] = stored
            while len(self.twins) > MAX_TWINS_IN_MEMORY:
                self.twins.pop(next(iter(self.twins)))
        self._log(stored)
        return self.summary(stored)

    def _log(self, stored: dict) -> None:
        CREATED_TWINS_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"twin_id": stored["id"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "label": stored["label"], "state": stored["state"], "offsets": stored["offsets"],
                 "built_from": [{"patient_id": n["entry"]["patient_id"], "line_index": n["entry"]["line_index"],
                                 "similarity": round(n["similarity"], 3)} for n in stored["neighbors"]]}
        with CREATED_TWINS_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")

    def get(self, twin_id: str) -> dict:
        stored = self.twins.get(twin_id)
        if stored is None:
            raise HTTPException(404, "Twin not found; upload the record again")
        return stored

    def fresh_twin(self, stored: dict) -> PatientDigitalTwin:
        return PatientDigitalTwin(stored["patient"], self.ensemble, self.toxicity, current_state=stored["state"],
                                  offsets=stored["offsets"])

    def regimen(self, key: str) -> list[str]:
        if key not in self.keys:
            raise HTTPException(400, f"Unknown medication {key!r}")
        return self.regimens[self.keys.index(key)]

    def summary(self, stored: dict) -> dict:
        s = stored["state"]
        pfi = s.get("months_since_last_platinum")
        platinum = ("No platinum given yet" if pfi is None else
                    "Platinum-resistant (relapse < 6 months after platinum)" if pfi < 6 else
                    "Partially platinum-sensitive (6-12 months)" if pfi < 12 else
                    "Platinum-sensitive (more than 12 months after platinum)")
        treatments = [{"line": l["line_index"] + 1, "start_month": round(l["start_day"] / 30.44, 1),
                       "classes": [CLASS_NAMES[c] for c in l["regimen"]]} for l in stored["patient"]["lines"]]
        position = self.project(s)
        return {
            "id": stored["id"], "label": stored["label"], "today_month": stored["today_month"],
            "map_position": [round(float(position[0]), 4), round(float(position[1]), 4)],
            "patient": {
                "age_now": round(s["age_years"], 1) if s.get("age_years") is not None else None,
                "stage": s.get("stage"), "grade": s.get("grade"), "residual": RESIDUAL_NAMES.get(s.get("residual")),
                "treatment_line": s["line_number"], "situation": SETTING_NAMES.get(s["setting"], s["setting"]),
                "last_tumor_status": {"TF": "Tumor-free", "WT": "Tumor present"}.get(s.get("last_tumor_status"),
                                                                                    "Not recorded"),
                "months_since_last_platinum": None if pfi is None else round(pfi, 1),
                "platinum_status": platinum, "past_treatments": treatments,
            },
            "build": {
                "database_size": len(self.database.entries),
                "neighbors": [{"rank": i + 1, "similarity": round(100 * n["similarity"]), "map_index": n["index"],
                               "age": n["entry"]["profile"]["age"], "stage": n["entry"]["profile"]["stage"],
                               "grade": n["entry"]["profile"]["grade"],
                               "residual": RESIDUAL_NAMES.get(n["entry"]["profile"]["residual"]),
                               "treatment_line": n["entry"]["profile"]["line_number"],
                               "situation": SETTING_NAMES.get(n["entry"]["profile"]["setting"]),
                               "prior_classes": [CLASS_NAMES[c] for c in n["entry"]["profile"]["prior_classes"]]}
                              for i, n in enumerate(stored["neighbors"])],
                "progression_risk_multiplier": round(math.exp(stored["offsets"]["pfs"]), 3),
            },
        }

    # --- predictions -----------------------------------------------------------------------
    def predict(self, stored: dict, key: str) -> dict:
        twin = self.fresh_twin(stored)
        p = twin.predict(self.regimen(key))
        allowed = bool(safety_mask(twin.state, self.regimens)[self.keys.index(key)])
        side = p["side_effects"]
        return {
            "medication": {"key": key, "name": MEDICATIONS[key][0], "group": MEDICATIONS[key][1]},
            "response_probability": p["response_probability"],
            "progression_free": p["progression_free_probability"],
            "expected_progression_free_months": p["expected_progression_free_months_5y"],
            "survival": p["survival_probability"],
            "expected_survival_months": p["expected_survival_months_5y"],
            "side_effects": {"any_severe": side["probability_any_severe"],
                             "by_organ": [{"organ": organ, "probability": prob, "level": side["levels"][organ]}
                                          for organ, prob in sorted(side["severe_event_probability"].items(),
                                                                    key=lambda kv: -kv[1]) if prob > 0]},
            "uncertainty": {"survival_months_std": p["uncertainty"]["survival_months_std"],
                            "progression_free_months_std": p["uncertainty"]["progression_free_months_std"],
                            "response_std": p["uncertainty"]["response_std"]},
            "allowed_by_safety_rules": allowed,
            "guideline_flags": guideline_flags(twin.state, self.regimen(key)),
        }

    def compare(self, stored: dict) -> list[dict]:
        rows = []
        for key in self.keys:
            p = self.predict(stored, key)
            rows.append({"key": key, "name": p["medication"]["name"], "group": p["medication"]["group"],
                         "response_probability": p["response_probability"],
                         "expected_progression_free_months": p["expected_progression_free_months"],
                         "expected_survival_months": p["expected_survival_months"],
                         "any_severe_side_effect": p["side_effects"]["any_severe"],
                         "survival_months_std": p["uncertainty"]["survival_months_std"],
                         "guideline_flags": p["guideline_flags"]})
        return sorted(rows, key=lambda r: -r["expected_survival_months"])

    def simulate(self, stored: dict, key: str, runs: int) -> dict:
        """Play the twin forward: `key` now, then standard care (guideline heuristic) at every later decision."""
        first = self.regimen(key)
        rng = random.Random(f"{stored['id']}:{key}")
        death_months, first_pf, lines, severe, journeys = [], [], [], [], []
        for run in range(runs):
            twin = self.fresh_twin(stored)
            journey, regimen = [], first
            while twin.end_reason is None and len(journey) < 8:
                outcome = twin.step(regimen, rng)
                journey.append({"medication": MEDICATIONS.get("+".join(regimen), ("+".join(regimen), ""))[0],
                                "months": round(outcome["months_on_line"], 1), "response": bool(outcome["response"]),
                                "severe_side_effects": outcome["severe_side_effects"],
                                "ended_with": "death" if outcome["died"] else
                                ("progression" if outcome["progressed"] else outcome["end_reason"])})
                if twin.end_reason is None:
                    mask = safety_mask(twin.state, self.regimens)
                    regimen = self.regimens[self.guideline.act(None, {"state": twin.state, "action_mask": mask})]
            death_months.append(twin.months_elapsed if twin.end_reason == "death" else math.inf)
            first_pf.append(journey[0]["months"] if journey[0]["ended_with"] in ("death", "progression") else math.inf)
            lines.append(len(journey))
            severe.append(sum(len(j["severe_side_effects"]) for j in journey))
            if run < 3:
                journeys.append(journey)
        months = list(range(0, int(HORIZON_MONTHS) + 1, 3))
        alive = [float(np.mean([d > m for d in death_months])) for m in months]
        progression_free = [float(np.mean([t > m for t in first_pf])) for m in months]
        median = next((m for m, a in zip(months, alive) if a < 0.5), None)
        return {"medication": MEDICATIONS[key][0], "runs": runs, "months": months, "alive": alive,
                "progression_free_first_line": progression_free, "median_survival_months": median,
                "alive_at_5_years": alive[-1], "mean_treatment_lines": float(np.mean(lines)),
                "mean_severe_side_effects": float(np.mean(severe)), "example_journeys": journeys,
                "later_lines_policy": "standard care (simplified guideline heuristic)"}

    def ai_doctor(self, stored: dict) -> dict:
        twin = self.fresh_twin(stored)
        safe = safety_mask(twin.state, self.regimens)
        allowed = [i for i in range(len(self.keys)) if safe[i] and not guideline_flags(twin.state, self.regimens[i])]
        excluded = [MEDICATIONS[k][0] for i, k in enumerate(self.keys) if i not in allowed]
        q = self.agent.q_values(build_observation(twin, self.toxicity.organs))
        ranked = sorted(allowed or range(len(self.keys)), key=lambda i: -q[i])
        return {"agent": "basic DQN (Objective 3 check)", "choice": MEDICATIONS[self.keys[ranked[0]]][0],
                "choice_key": self.keys[ranked[0]],
                "ranking": [{"key": self.keys[i], "name": MEDICATIONS[self.keys[i]][0], "value": float(q[i])}
                            for i in ranked[:5]],
                "excluded_by_constraints": excluded,
                "note": ("Experimental agent trained only inside simulated twins. It scores whole 5-year treatment "
                         "journeys (survival, response, side effects), so its choice can differ from the single-line "
                         "ranking. Not medical advice.")}


def create_app(state: DemoState | None = None) -> FastAPI:
    app = FastAPI(title="Ovarian Cancer Digital Twin - mid-semester demo")
    demo = state or DemoState()

    @app.get("/api/info")
    def info():
        return demo.info

    @app.post("/api/twins")
    async def upload(file: UploadFile = File(...)):
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large for a patient record")
        try:
            record = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HTTPException(422, f"Not a valid JSON file: {error}") from error
        try:
            return demo.create(record)
        except RecordError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/twins/{twin_id}")
    def twin_summary(twin_id: str):
        return demo.summary(demo.get(twin_id))

    @app.post("/api/twins/{twin_id}/predict")
    def predict(twin_id: str, request: MedicationRequest):
        return demo.predict(demo.get(twin_id), request.medication)

    @app.get("/api/twins/{twin_id}/compare")
    def compare(twin_id: str):
        return demo.compare(demo.get(twin_id))

    @app.post("/api/twins/{twin_id}/simulate")
    def simulate(twin_id: str, request: SimulationRequest):
        return demo.simulate(demo.get(twin_id), request.medication, request.runs)

    @app.get("/api/twins/{twin_id}/ai-doctor")
    def ai_doctor(twin_id: str):
        return demo.ai_doctor(demo.get(twin_id))

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/")
        def missing_frontend():
            return JSONResponse({"message": "Website not built yet: run `pnpm install && pnpm build` in demo/web"})
    return app
