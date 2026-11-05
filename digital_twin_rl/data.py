"""Step 1: turn the TCGA-OV clinical export into one treatment history per patient.

A patient's dated drug and radiation records that start within 6 weeks of each other form one
treatment line. For every line we keep:

    state    what was known just before the line: age, stage, grade, tumour left after surgery,
             line number, disease setting, drug classes given before, time since platinum
    regimen  the drug classes given, e.g. ["platinum", "taxanes"]
    outcome  tumour response, progression-free survival and overall survival
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DAYS_PER_MONTH = 30.44
LINE_WINDOW_DAYS = 42             # records starting within 6 weeks form one treatment line
RESPONSE_WINDOW_DAYS = (30, 548)  # a check-up 1 to 18 months after the line start is its response
PROGRESSION_MIN_DAYS = 30         # progression dated at the line start itself is ignored

DRUG_CLASSES = ["platinum", "taxanes", "anthracyclines", "antimetabolites", "topoisomerase_inhibitors",
                "alkylating_agents", "anti_angiogenics", "hormonal_agents", "vinca_alkaloids", "radiation", "other"]
SETTINGS = ["primary", "recurrent", "progressive", "unknown"]
AGENT_CLASS = {
    "carboplatin": "platinum", "cisplatin": "platinum", "oxaliplatin": "platinum",
    "paclitaxel": "taxanes", "docetaxel": "taxanes", "nab-paclitaxel": "taxanes",
    "paclitaxel poliglumex": "taxanes", "taxane compound": "taxanes",
    "pegylated liposomal doxorubicin hydrochloride": "anthracyclines",
    "doxorubicin": "anthracyclines", "doxorubicin hydrochloride": "anthracyclines",
    "gemcitabine": "antimetabolites", "gemcitabine hydrochloride": "antimetabolites",
    "capecitabine": "antimetabolites", "pemetrexed disodium": "antimetabolites",
    "pemetrexed": "antimetabolites", "fluorouracil": "antimetabolites",
    "aminopterin": "antimetabolites", "plevitrexed": "antimetabolites", "leucovorin": "antimetabolites",
    "topotecan": "topoisomerase_inhibitors", "topotecan hydrochloride": "topoisomerase_inhibitors",
    "etoposide": "topoisomerase_inhibitors", "irinotecan hydrochloride": "topoisomerase_inhibitors",
    "aminocamptothecin": "topoisomerase_inhibitors", "vosaroxin": "topoisomerase_inhibitors",
    "cyclophosphamide": "alkylating_agents", "altretamine": "alkylating_agents",
    "ifosfamide": "alkylating_agents", "melphalan": "alkylating_agents",
    "chlorambucil": "alkylating_agents", "trabectedin": "alkylating_agents", "irofulven": "alkylating_agents",
    "bevacizumab": "anti_angiogenics", "motesanib diphosphate": "anti_angiogenics",
    "pan-vegfr/tie2 tyrosine kinase inhibitor cep-11981": "anti_angiogenics", "sorafenib": "anti_angiogenics",
    "tamoxifen": "hormonal_agents", "letrozole": "hormonal_agents", "anastrozole": "hormonal_agents",
    "megestrol acetate": "hormonal_agents", "fulvestrant": "hormonal_agents", "exemestane": "hormonal_agents",
    "leuprolide acetate": "hormonal_agents", "medroxyprogesterone acetate": "hormonal_agents",
    "vinorelbine tartrate": "vinca_alkaloids", "vinorelbine": "vinca_alkaloids", "vincristine": "vinca_alkaloids",
}
UNKNOWN = {"", "unknown", "not reported", "not available", "not applicable"}
STAGE_LEVELS = {"I": 1, "II": 2, "III": 3, "IV": 4}
GRADE_LEVELS = {"G1": 1, "G2": 2, "G3": 3, "G4": 3}
RESIDUAL_LEVELS = {"No macroscopic disease": 0, "1-10 mm": 1, "11-20 mm": 2, ">20 mm": 3}
STATUS_TO_SETTING = {"recurrent disease": "recurrent", "progressive disease": "progressive",
                     "persistent disease": "progressive"}


def clean(value) -> str:
    return str(value).strip() if value is not None else ""


def known(value) -> bool:
    return clean(value).lower() not in UNKNOWN


def as_day(value) -> int | None:
    """Whole-number day offsets only; anything else counts as missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def months(days: float) -> float:
    return days / DAYS_PER_MONTH


def drug_classes(treatment: dict) -> list[str]:
    """Drug classes of one treatment record; [] for surgery, a therapy marked as not given, or no agent."""
    if clean(treatment.get("treatment_or_therapy")).lower() == "no":
        return []
    kind = clean(treatment.get("treatment_type")).lower()
    if "radiation" in kind or "radioisotope" in kind:
        return ["radiation"]
    if "surgery" in kind or not known(treatment.get("therapeutic_agents")):
        return []
    names = [n.strip().lower() for n in clean(treatment["therapeutic_agents"]).replace(";", ",").split(",") if n.strip()]
    return sorted({AGENT_CLASS.get(name, "other") for name in names})


def dated_treatments(case: dict) -> list[dict]:
    records = []
    for diagnosis in case.get("diagnoses") or []:
        for treatment in diagnosis.get("treatments") or []:
            classes = drug_classes(treatment)
            start = as_day(treatment.get("days_to_treatment_start"))
            if not classes or start is None or start < 0:
                continue
            end = as_day(treatment.get("days_to_treatment_end"))
            records.append({"start": start, "end": end if end is not None and end >= start else None,
                            "classes": classes,
                            "setting": STATUS_TO_SETTING.get(clean(treatment.get("initial_disease_status")).lower())})
    return sorted(records, key=lambda r: r["start"])


def group_lines(records: list[dict]) -> list[dict]:
    lines: list[dict] = []
    for record in records:
        if lines and record["start"] <= lines[-1]["start"] + LINE_WINDOW_DAYS:
            lines[-1]["records"].append(record)
        else:
            lines.append({"start": record["start"], "records": [record]})
    for index, line in enumerate(lines):
        line["classes"] = sorted({c for r in line["records"] for c in r["classes"]})
        ends = [r["end"] for r in line["records"] if r["end"] is not None]
        line["end"] = max(ends) if ends else None
        settings = {r["setting"] for r in line["records"]}
        line["setting"] = ("progressive" if "progressive" in settings else "recurrent" if "recurrent" in settings
                           else "primary" if index == 0 else "unknown")
    return lines


def response_timeline(case: dict) -> list[tuple[int, str]]:
    """(day, recorded tumour status) for check-ups with exactly one status that day."""
    by_day = defaultdict(set)
    for follow_up in case.get("follow_ups") or []:
        day = as_day(follow_up.get("days_to_follow_up"))
        if day is not None and day >= 0 and known(follow_up.get("disease_response")):
            by_day[day].add(clean(follow_up["disease_response"]))
    return [(day, next(iter(labels))) for day, labels in sorted(by_day.items()) if len(labels) == 1]


def progression_days(case: dict) -> list[int]:
    days = {as_day(f.get(field)) for f in case.get("follow_ups") or []
            for field in ("days_to_progression", "days_to_recurrence")}
    return sorted(d for d in days if d is not None and d >= 0)


def last_contact_day(case: dict, records: list[dict]) -> int:
    days = [0] + [r["end"] or r["start"] for r in records]
    days += [as_day(d.get("days_to_last_follow_up")) for d in case.get("diagnoses") or []]
    days += [as_day(f.get("days_to_follow_up")) for f in case.get("follow_ups") or []]
    days.append(as_day((case.get("demographic") or {}).get("days_to_death")))
    return max(d for d in days if d is not None)


def patient_static(case: dict) -> dict:
    primary = next((d for d in case.get("diagnoses") or []
                    if clean(d.get("diagnosis_is_primary_disease")).lower() == "true"
                    or clean(d.get("classification_of_tumor")).lower() == "primary"), {})
    stage = "".join(ch for ch in clean(primary.get("figo_stage")).upper().replace("STAGE", "").strip() if ch in "IV")
    residual = next((RESIDUAL_LEVELS[clean(p.get("residual_tumor_measurement"))]
                     for p in primary.get("pathology_details") or []
                     if clean(p.get("residual_tumor_measurement")) in RESIDUAL_LEVELS), None)
    try:
        age = float((case.get("demographic") or {}).get("age_at_index"))
    except (TypeError, ValueError):
        age = None
    return {"age_at_diagnosis": age, "stage": STAGE_LEVELS.get(stage),
            "grade": GRADE_LEVELS.get(clean(primary.get("tumor_grade"))), "residual": residual}


def pre_line_state(static: dict, previous: list[dict], day: int, setting: str) -> dict:
    """What is known just before a treatment line that starts on `day` (days since diagnosis)."""
    exposure = Counter(c for line in previous for c in line["classes"])
    platinum = [line for line in previous if "platinum" in line["classes"]]
    last_platinum = (platinum[-1]["end"] or platinum[-1]["start"]) if platinum else None
    last_line = (previous[-1]["end"] or previous[-1]["start"]) if previous else None
    age = static["age_at_diagnosis"]
    return {
        "age_years": age + day / 365.25 if age is not None else None,
        "stage": static["stage"], "grade": static["grade"], "residual": static["residual"],
        "line_number": len(previous) + 1,
        "months_since_diagnosis": months(day),
        "setting": setting,
        "prior_exposure": {c: exposure[c] for c in DRUG_CLASSES if exposure[c]},
        "months_since_last_platinum": months(max(0, day - last_platinum)) if last_platinum is not None else None,
        "months_since_last_line": months(max(0, day - last_line)) if last_line is not None else None,
    }


def state_at(case: dict, day: int, setting: str) -> dict:
    """State for a new treatment decision on `day`, e.g. 'today' in an uploaded record."""
    previous = [line for line in group_lines(dated_treatments(case)) if line["start"] < day]
    return pre_line_state(patient_static(case), previous, day, setting)


def build_patient(case: dict, split: str) -> dict:
    demographic = case.get("demographic") or {}
    records = dated_treatments(case)
    lines = group_lines(records)
    timeline = response_timeline(case)
    progressions = progression_days(case)
    last_day = last_contact_day(case, records)
    death_day = as_day(demographic.get("days_to_death"))
    if death_day is None and clean(demographic.get("vital_status")).lower() == "dead":
        death_day = last_day        # recorded dead without a date: the death happened at last contact
    static = patient_static(case)

    built = []
    for index, line in enumerate(lines):
        start = line["start"]
        next_start = lines[index + 1]["start"] if index + 1 < len(lines) else None
        window_end = min(start + RESPONSE_WINDOW_DAYS[1], next_start or start + RESPONSE_WINDOW_DAYS[1])
        response = next((label for day, label in timeline if start + RESPONSE_WINDOW_DAYS[0] <= day <= window_end), None)
        # First event after the line: recorded progression/recurrence, the next line starting, or death.
        events = [(d, "progression") for d in progressions if d > start + PROGRESSION_MIN_DAYS]
        if next_start is not None:
            events.append((next_start, "next_line"))
        died = death_day is not None and death_day >= start
        if died:
            events.append((death_day, "death"))
        first = min(events, key=lambda e: (e[0], e[1] == "death")) if events else None
        built.append({
            "line_index": index, "start_day": start, "end_day": line["end"], "regimen": line["classes"],
            "state": pre_line_state(static, lines[:index], start, line["setting"]),
            "outcome": {
                "response": 1 if response == "TF-Tumor Free" else 0 if response == "WT-With Tumor" else None,
                "pfs_months": months((first[0] if first else last_day) - start),
                "pfs_event": int(first is not None),
                "pfs_event_is_death": int(first[1] == "death") if first else None,
                "os_months": months((death_day if died else last_day) - start),
                "os_event": int(died),
            },
        })
    return {"patient_id": str(case.get("case_id") or case.get("submitter_id")), "split": split,
            "static": static, "lines": built}


def assign_splits(patient_ids: list[str], seed: int = 42) -> dict[str, str]:
    """80% train, 10% validation, 10% test, by patient."""
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    n_train, n_val = int(len(ids) * 0.8), round(len(ids) * 0.1)
    return {pid: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
            for i, pid in enumerate(ids)}


def build_histories(cases: list[dict]) -> list[dict]:
    """One history per patient whose time axis starts at diagnosis."""
    eligible = [c for c in cases if c.get("index_date") == "Diagnosis"]
    splits = assign_splits([str(c.get("case_id") or c.get("submitter_id")) for c in eligible])
    return [build_patient(c, splits[str(c.get("case_id") or c.get("submitter_id"))]) for c in eligible]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
