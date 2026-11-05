"""The patient file uploaded on the website (format: demo/sample_patients/FORMAT.md).

The record is converted to the dataset's own case format and encoded by the same code as the
training data, so an uploaded patient is described exactly like the patients the model learned from.
"""
from __future__ import annotations

from .data import AGENT_CLASS, DAYS_PER_MONTH, group_lines, dated_treatments, state_at

# ---------------------------------------------------------------------------------------------
# Allowed values in an uploaded record, and how they map to the dataset's own wording.
# ---------------------------------------------------------------------------------------------
FORMAT = "ovarian-twin-record/v1"
STAGES = ["I", "IA", "IB", "IC", "II", "IIA", "IIB", "IIC", "III", "IIIA", "IIIB", "IIIC", "IV"]
GRADES = ["G1", "G2", "G3", "G4"]
RESIDUAL = {"none": "No macroscopic disease", "1-10 mm": "1-10 mm", "11-20 mm": "11-20 mm", ">20 mm": ">20 mm"}
SITUATIONS = {"newly diagnosed": "primary", "recurrence": "recurrent", "progression": "progressive", "unknown": "unknown"}
REASONS = {"first treatment": None, "recurrence": "Recurrent Disease", "progression": "Progressive Disease"}
EVENTS = ["recurrence", "progression"]
# Drug names accepted in a record (lower case -> display name): every drug in the dataset, plus radiation.
DRUGS = {name: name.title() for name in AGENT_CLASS} | {"radiation": "Radiation"}


class RecordError(ValueError):
    """The uploaded file is not a valid record; the message says what to fix."""


# ---------------------------------------------------------------------------------------------
# Validation helpers: each one raises a RecordError with a clear message when a value is wrong.
# ---------------------------------------------------------------------------------------------
def _number(value, field: str, low: float, high: float) -> float:
    # A number between low and high.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"'{field}' must be a number")
    if not low <= value <= high:
        raise RecordError(f"'{field}' must be between {low:g} and {high:g} (got {value})")
    return float(value)


def _choice(value, field: str, options) -> str:
    # One of the allowed options, ignoring upper/lower case.
    for option in options:
        if str(value).strip().lower() == option.lower():
            return option
    raise RecordError(f"'{field}' must be one of: {', '.join(options)} (got {value!r})")


def _day(months: float) -> int:
    # Records use months since diagnosis; the dataset uses days.
    return round(months * DAYS_PER_MONTH)


# ---------------------------------------------------------------------------------------------
# Parsing: uploaded record -> the patient's state today.
# ---------------------------------------------------------------------------------------------
def parse_record(record: dict) -> dict:
    """Validate an uploaded record; return the patient's state today and their past treatment lines."""
    # Basic checks and the facts from diagnosis.
    if not isinstance(record, dict):
        raise RecordError("The file must contain one JSON object")
    if record.get("format", FORMAT) != FORMAT:
        raise RecordError(f"Unsupported format {record.get('format')!r}; expected {FORMAT!r}")
    today = _number(record.get("today_month"), "today_month", 0, 600)
    age = _number(record.get("age_at_diagnosis"), "age_at_diagnosis", 18, 100)
    stage = _choice(str(record.get("figo_stage", "")).upper().replace("STAGE", "").strip(), "figo_stage", STAGES)
    grade = _choice(record["tumor_grade"], "tumor_grade", GRADES) if record.get("tumor_grade") else None
    residual = record.get("residual_tumor_after_surgery")
    residual = RESIDUAL[_choice(residual, "residual_tumor_after_surgery", list(RESIDUAL))] if residual else None

    # Past treatments: check each course, then write one dataset-style treatment record per drug.
    treatments = []
    for i, entry in enumerate(record.get("treatments") or []):
        where = f"treatments[{i}]"
        if not isinstance(entry.get("drugs"), list) or not entry["drugs"]:
            raise RecordError(f"{where}.drugs must be a non-empty list of drug names")
        start = _number(entry.get("start_month"), f"{where}.start_month", 0, today)
        end = None if entry.get("end_month") is None else _number(entry["end_month"], f"{where}.end_month", start, today)
        reason = REASONS[_choice(entry["reason"], f"{where}.reason", list(REASONS))] if entry.get("reason") else None
        for drug in entry["drugs"]:
            name = DRUGS.get(str(drug).strip().lower())
            if name is None:
                raise RecordError(f"{where}: unknown drug {drug!r}. Known drugs: {', '.join(sorted(DRUGS.values()))}")
            radiation = name == "Radiation"
            treatments.append({"therapeutic_agents": None if radiation else name,
                               "treatment_type": "Radiation Therapy, NOS" if radiation else "Chemotherapy",
                               "initial_disease_status": reason, "days_to_treatment_start": _day(start),
                               "days_to_treatment_end": None if end is None else _day(end)})

    # Check-ups: keep the recurrence/progression events (used below to work out the situation).
    events = []
    for i, entry in enumerate(record.get("follow_ups") or []):
        month = _number(entry.get("month"), f"follow_ups[{i}].month", 0, today)
        if entry.get("event"):
            events.append((month, _choice(entry["event"], f"follow_ups[{i}].event", EVENTS)))

    # Today's situation: as stated in the record; else "newly diagnosed" if there were no
    # treatments; else the latest recorded event; else unknown.
    if record.get("current_situation"):
        setting = SITUATIONS[_choice(record["current_situation"], "current_situation", list(SITUATIONS))]
    elif not treatments:
        setting = "primary"
    else:                               # the latest recorded event, if any
        setting = SITUATIONS[max(events)[1]] if events else "unknown"

    # Build a case in the dataset's own format, then reuse the training-data code to work out
    # the treatment lines and today's state.
    case = {"index_date": "Diagnosis", "demographic": {"age_at_index": age},
            "diagnoses": [{"diagnosis_is_primary_disease": "true", "figo_stage": f"Stage {stage}", "tumor_grade": grade,
                           "pathology_details": [{"residual_tumor_measurement": residual}] if residual else [],
                           "treatments": treatments}]}
    today_day = _day(today)
    past = [line for line in group_lines(dated_treatments(case)) if line["start"] < today_day]
    return {"label": str(record.get("patient_label") or "Uploaded patient"), "today_month": round(today, 2),
            "state": state_at(case, today_day, setting),
            "past_lines": [{"start_month": round(line["start"] / DAYS_PER_MONTH, 1), "classes": line["classes"]}
                           for line in past]}
