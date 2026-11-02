"""Simple patient-record format for the demo, converted to and from the dataset's case format.

A record describes one ovarian-cancer patient up to "today", with times in
months since diagnosis. `parse_record` turns it into a GDC-style case and runs
it through the same code as the training data (`build_patient`, `state_at`),
so an uploaded patient is encoded exactly like the patients the twin learned
from. `case_to_record` does the reverse for real dataset patients, cut off at
one of their decision points (used to make the demo files).

See demo/sample_patients/FORMAT.md for the field reference.
"""
from __future__ import annotations

import re

from scripts.paper_method_pipeline import AGENT_CLASS, as_day, clean, classify_treatment, is_known

from .data import DAYS_PER_MONTH, RESIDUAL_LEVELS, build_patient, state_at

FORMAT = "ovarian-twin-record/v1"
STAGES = ["I", "IA", "IB", "IC", "II", "IIA", "IIB", "IIC", "III", "IIIA", "IIIB", "IIIC", "IV"]
GRADES = ["G1", "G2", "G3", "G4"]
RESIDUAL = {"none": "No macroscopic disease", "1-10 mm": "1-10 mm", "11-20 mm": "11-20 mm", ">20 mm": ">20 mm"}
SITUATIONS = {"newly diagnosed": "primary", "recurrence": "recurrent", "progression": "progressive", "unknown": "unknown"}
REASONS = {"first treatment": None, "recurrence": "Recurrent Disease", "progression": "Progressive Disease"}
TUMOR_STATUS = {"tumor-free": "TF-Tumor Free", "tumor present": "WT-With Tumor"}
EVENTS = {"recurrence": "days_to_recurrence", "progression": "days_to_progression"}
RADIATION = "Radiation"
KNOWN_DRUGS = sorted({name.title() for name in AGENT_CLASS} | {RADIATION})
_DRUG_LOOKUP = {name.lower(): name for name in KNOWN_DRUGS}


class RecordError(ValueError):
    """The uploaded record is not in the expected format; the message says what to fix."""


def _day(months: float) -> int:
    return int(round(float(months) * DAYS_PER_MONTH))


def _month(day: int) -> float:
    return round(day / DAYS_PER_MONTH, 2)


def _number(value, field: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"'{field}' must be a number")
    if not low <= value <= high:
        raise RecordError(f"'{field}' must be between {low:g} and {high:g} (got {value})")
    return float(value)


def _choice(value, field: str, options) -> str:
    text = str(value).strip()
    for option in options:
        if text.lower() == option.lower():
            return option
    raise RecordError(f"'{field}' must be one of: {', '.join(options)} (got {value!r})")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "patient"


def record_to_case(record: dict) -> tuple[dict, int, str | None]:
    """Validate a record and build the equivalent GDC-style case; returns (case, today_day, setting)."""
    if not isinstance(record, dict):
        raise RecordError("The file must contain one JSON object")
    if record.get("format", FORMAT) != FORMAT:
        raise RecordError(f"Unsupported format {record.get('format')!r}; expected {FORMAT!r}")
    today = _number(record.get("today_month"), "today_month", 0, 600)
    age = _number(record.get("age_at_diagnosis"), "age_at_diagnosis", 18, 100)
    stage = _choice(str(record.get("figo_stage", "")).upper().replace("STAGE", "").strip(), "figo_stage", STAGES)
    grade = _choice(record["tumor_grade"], "tumor_grade", GRADES) if record.get("tumor_grade") else None
    residual = RESIDUAL[_choice(record["residual_tumor_after_surgery"], "residual_tumor_after_surgery", list(RESIDUAL))] \
        if record.get("residual_tumor_after_surgery") else None

    treatments = []
    for i, entry in enumerate(record.get("treatments") or []):
        where = f"treatments[{i}]"
        drugs = entry.get("drugs")
        if not isinstance(drugs, list) or not drugs:
            raise RecordError(f"{where}.drugs must be a non-empty list of drug names")
        start = _number(entry.get("start_month"), f"{where}.start_month", 0, today)
        end = entry.get("end_month")
        end = None if end is None else _number(end, f"{where}.end_month", start, today)
        reason = REASONS[_choice(entry["reason"], f"{where}.reason", list(REASONS))] if entry.get("reason") else None
        for drug in drugs:
            name = _DRUG_LOOKUP.get(str(drug).strip().lower())
            if name is None:
                raise RecordError(f"{where}: unknown drug {drug!r}. Known drugs: {', '.join(KNOWN_DRUGS)}")
            treatments.append({
                "therapeutic_agents": None if name == RADIATION else name,
                "treatment_type": "Radiation Therapy, NOS" if name == RADIATION else "Chemotherapy",
                "treatment_or_therapy": "yes", "initial_disease_status": reason,
                "days_to_treatment_start": _day(start), "days_to_treatment_end": None if end is None else _day(end),
            })

    follow_ups = []
    for i, entry in enumerate(record.get("follow_ups") or []):
        where = f"follow_ups[{i}]"
        day = _day(_number(entry.get("month"), f"{where}.month", 0, today))
        row = {"days_to_follow_up": day}
        if entry.get("tumor_status"):
            row["disease_response"] = TUMOR_STATUS[_choice(entry["tumor_status"], f"{where}.tumor_status", list(TUMOR_STATUS))]
        if entry.get("event"):
            row[EVENTS[_choice(entry["event"], f"{where}.event", list(EVENTS))]] = day
        follow_ups.append(row)

    setting = None
    if record.get("current_situation"):
        setting = SITUATIONS[_choice(record["current_situation"], "current_situation", list(SITUATIONS))]
    elif not treatments:
        setting = "primary"
    else:
        events = sorted((row["days_to_follow_up"], key) for row in follow_ups for key in EVENTS if EVENTS[key] in row)
        setting = SITUATIONS[events[-1][1]] if events else "unknown"

    case = {
        "case_id": f"uploaded-{_slug(str(record.get('patient_label') or 'patient'))}",
        "index_date": "Diagnosis",
        "demographic": {"age_at_index": age, "vital_status": "Alive"},
        "diagnoses": [{"diagnosis_is_primary_disease": "true", "classification_of_tumor": "primary",
                       "figo_stage": f"Stage {stage}", "tumor_grade": grade,
                       "pathology_details": [{"residual_tumor_measurement": residual}] if residual else [],
                       "treatments": treatments}],
        "follow_ups": follow_ups,
    }
    return case, _day(today), setting


def parse_record(record: dict) -> dict:
    """Uploaded record -> patient history (as in the dataset) and the pre-treatment state for today."""
    case, today_day, setting = record_to_case(record)
    patient = build_patient(case, "uploaded")
    patient["lines"] = [line for line in patient["lines"] if line["start_day"] < today_day]
    state = state_at(case, today_day, setting)
    return {"label": str(record.get("patient_label") or "Uploaded patient"), "patient": patient,
            "state": state, "today_month": _month(today_day)}


def case_to_record(case: dict, decision_day: int, label: str, setting: str | None = None) -> dict:
    """A real dataset case, cut off just before `decision_day`, in the simple record format."""
    diagnosis = next((d for d in case.get("diagnoses") or []
                      if clean(d.get("diagnosis_is_primary_disease")).lower() == "true"
                      or clean(d.get("classification_of_tumor")).lower() == "primary"), {})
    residual = next((detail.get("residual_tumor_measurement") for detail in diagnosis.get("pathology_details") or []
                     if detail.get("residual_tumor_measurement") in RESIDUAL_LEVELS), None)
    groups: dict[tuple, dict] = {}
    for d in case.get("diagnoses") or []:
        for t in d.get("treatments") or []:
            if clean(t.get("treatment_or_therapy")).lower() == "no":
                continue
            classes, _ = classify_treatment(t)
            start, end = as_day(t.get("days_to_treatment_start")), as_day(t.get("days_to_treatment_end"))
            if not classes or start is None or start < 0 or start >= decision_day:
                continue
            if classes == ["radiation"]:
                drugs = [RADIATION]
            else:
                drugs = [s.strip().title() for s in clean(t.get("therapeutic_agents")).replace(";", ",").split(",") if s.strip()]
            unknown = [drug for drug in drugs if drug.lower() not in _DRUG_LOOKUP]
            if unknown:
                raise RecordError(f"Drugs {unknown} are not in the record vocabulary")
            end = end if end is not None and start <= end < decision_day else None
            status = clean(t.get("initial_disease_status")).lower()
            reason = "recurrence" if status == "recurrent disease" else "progression" if status in (
                "progressive disease", "persistent disease") else None
            group = groups.setdefault((start, end, reason), {"drugs": [], "start_month": _month(start)})
            if end is not None:
                group["end_month"] = _month(end)
            if reason:
                group["reason"] = reason
            group["drugs"].extend(drug for drug in drugs if drug not in group["drugs"])
    follow_ups = []
    for f in case.get("follow_ups") or []:
        day = as_day(f.get("days_to_follow_up"))
        if day is not None and 0 <= day < decision_day and is_known(f.get("disease_response")):
            status = {v: k for k, v in TUMOR_STATUS.items()}.get(clean(f.get("disease_response")))
            if status:
                follow_ups.append({"month": _month(day), "tumor_status": status})
        for event, field in EVENTS.items():
            event_day = as_day(f.get(field))
            if event_day is not None and 0 <= event_day < decision_day:
                follow_ups.append({"month": _month(event_day), "event": event})
    unique_follow_ups = {(f["month"], f.get("tumor_status"), f.get("event")): f for f in follow_ups}
    record = {
        "format": FORMAT,
        "patient_label": label,
        "age_at_diagnosis": (case.get("demographic") or {}).get("age_at_index"),
        "figo_stage": clean(diagnosis.get("figo_stage")).replace("Stage ", ""),
        "tumor_grade": diagnosis.get("tumor_grade") if diagnosis.get("tumor_grade") in GRADES else None,
        "residual_tumor_after_surgery": {v: k for k, v in RESIDUAL.items()}.get(residual),
        "treatments": sorted(groups.values(), key=lambda g: g["start_month"]),
        "follow_ups": sorted(unique_follow_ups.values(), key=lambda f: f["month"]),
        "today_month": _month(decision_day),
    }
    if setting is not None:
        record["current_situation"] = {v: k for k, v in SITUATIONS.items()}[setting]
    return record
