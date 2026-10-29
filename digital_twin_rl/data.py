"""Build per-patient treatment-line histories from the TCGA-OV clinical export.

Each eligible patient (index_date = Diagnosis) becomes an ordered list of
treatment lines. A line groups dated drug/radiation records that start within
LINE_WINDOW_DAYS of the line's first record. For every line we keep the
pre-treatment state, the regimen (the paper's 11 classes), and the observed
outcomes: tumor response, progression-free interval and overall survival.

Records with treatment_or_therapy = "no" document a therapy that was NOT given
(931 "Radiation Therapy, NOS" rows in the 2026-10-09 export). They are excluded
from actions and from the undated-history flag.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from scripts.paper_method_pipeline import (
    ACTION_CLASSES, as_day, as_number, assign_splits, choose_primary_diagnosis,
    classify_treatment, clean, response_timeline,
)

DAYS_PER_MONTH = 30.44
LINE_WINDOW_DAYS = 42          # records starting within 6 weeks form one regimen
RESPONSE_MIN_DAYS = 30         # earliest follow-up counted as a response to the line
RESPONSE_MAX_DAYS = 548        # latest (about 18 months) follow-up counted
PROGRESSION_MIN_DAYS = 30      # ignore progression dated at the line start itself
SETTINGS = ["primary", "recurrent", "progressive", "unknown"]
RESIDUAL_LEVELS = {"No macroscopic disease": 0, "1-10 mm": 1, "11-20 mm": 2, ">20 mm": 3}
GRADE_LEVELS = {"G1": 1, "G2": 2, "G3": 3, "G4": 3}
STATUS_TO_SETTING = {
    "recurrent disease": "recurrent",
    "progressive disease": "progressive",
    "persistent disease": "progressive",
}


def months(days: float) -> float:
    return days / DAYS_PER_MONTH


def stage_number(figo: str | None) -> int | None:
    """'Stage IIIC' -> 3, 'Stage IV' -> 4; None when absent or unparseable."""
    text = clean(figo).upper().replace("STAGE", "").strip()
    roman = "".join(ch for ch in text if ch in "IV")
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(roman)


def residual_level(diagnosis: dict) -> int | None:
    for detail in diagnosis.get("pathology_details") or []:
        level = RESIDUAL_LEVELS.get(clean(detail.get("residual_tumor_measurement")))
        if level is not None:
            return level
    return None


def given_treatments(case: dict) -> tuple[list[dict], int]:
    """Return dated, classifiable given treatments and the count of undated given ones."""
    dated, undated_given = [], 0
    for diagnosis in case.get("diagnoses") or []:
        for treatment in diagnosis.get("treatments") or []:
            if clean(treatment.get("treatment_or_therapy")).lower() == "no":
                continue
            classes, _ = classify_treatment(treatment)
            if not classes:
                continue
            start = as_day(treatment.get("days_to_treatment_start"))
            if start is None or start < 0:
                undated_given += 1
                continue
            end = as_day(treatment.get("days_to_treatment_end"))
            dated.append({
                "start": start,
                "end": end if end is not None and end >= start else None,
                "classes": classes,
                "setting": STATUS_TO_SETTING.get(clean(treatment.get("initial_disease_status")).lower()),
            })
    dated.sort(key=lambda m: m["start"])
    return dated, undated_given


def group_lines(members: list[dict]) -> list[dict]:
    lines: list[dict] = []
    for member in members:
        if lines and member["start"] <= lines[-1]["start"] + LINE_WINDOW_DAYS:
            lines[-1]["members"].append(member)
        else:
            lines.append({"start": member["start"], "members": [member]})
    for index, line in enumerate(lines):
        line["classes"] = sorted({c for m in line["members"] for c in m["classes"]})
        ends = [m["end"] for m in line["members"] if m["end"] is not None]
        line["end"] = max(ends) if ends else None
        settings = {m["setting"] for m in line["members"] if m["setting"]}
        if "progressive" in settings:
            line["setting"] = "progressive"
        elif "recurrent" in settings:
            line["setting"] = "recurrent"
        else:
            line["setting"] = "primary" if index == 0 else "unknown"
    return lines


def progression_days(case: dict) -> list[int]:
    days = set()
    for follow_up in case.get("follow_ups") or []:
        for field in ("days_to_progression", "days_to_recurrence"):
            day = as_day(follow_up.get(field))
            if day is not None and day >= 0:
                days.add(day)
    return sorted(days)


def last_contact_day(case: dict, members: list[dict]) -> int:
    days = [0]
    for diagnosis in case.get("diagnoses") or []:
        day = as_day(diagnosis.get("days_to_last_follow_up"))
        if day is not None:
            days.append(day)
    for follow_up in case.get("follow_ups") or []:
        day = as_day(follow_up.get("days_to_follow_up"))
        if day is not None:
            days.append(day)
    days.extend(m["end"] or m["start"] for m in members)
    death = as_day((case.get("demographic") or {}).get("days_to_death"))
    if death is not None:
        days.append(death)
    return max(days)


def latest_known(timeline: list[dict], day: int, field: str):
    for point in reversed(timeline):
        if point["day"] < day and point.get(field) is not None:
            return point[field]
    return None


def build_patient(case: dict, split: str) -> dict:
    patient_id = str(case.get("case_id") or case.get("submitter_id"))
    demographic = case.get("demographic") or {}
    primary = choose_primary_diagnosis(case)
    age = as_number(demographic.get("age_at_index"))
    members, undated_given = given_treatments(case)
    lines = group_lines(members)
    timeline = response_timeline(case)
    progressions = progression_days(case)
    death_day = as_day(demographic.get("days_to_death"))
    last_day = last_contact_day(case, members)
    death_day_imputed = death_day is None and clean(demographic.get("vital_status")).lower() == "dead"
    if death_day_imputed:
        death_day = last_day    # recorded dead without a death day: event at last contact, not censoring
    static = {
        "age_at_diagnosis": age,
        "stage": stage_number(primary.get("figo_stage")),
        "grade": GRADE_LEVELS.get(clean(primary.get("tumor_grade"))),
        "residual": residual_level(primary),
        "undated_history": int(undated_given > 0),
    }

    built = []
    for index, line in enumerate(lines):
        start = line["start"]
        previous = lines[:index]
        exposure = Counter(c for p in previous for c in p["classes"])
        platinum_lines = [p for p in previous if "platinum" in p["classes"]]
        last_platinum = (platinum_lines[-1]["end"] or platinum_lines[-1]["start"]) if platinum_lines else None
        prev_end = (previous[-1]["end"] or previous[-1]["start"]) if previous else None
        tumor = latest_known(timeline, start, "response")
        state = {
            "age_years": age + start / 365.25 if age is not None else None,
            "stage": static["stage"],
            "grade": static["grade"],
            "residual": static["residual"],
            "line_number": index + 1,
            "months_since_diagnosis": months(start),
            "setting": line["setting"],
            "last_tumor_status": "TF" if tumor == "TF-Tumor Free" else "WT" if tumor == "WT-With Tumor" else None,
            "ecog": as_number(latest_known(timeline, start, "ecog")),
            "karnofsky": as_number(latest_known(timeline, start, "karnofsky")),
            "prior_exposure": {c: exposure[c] for c in ACTION_CLASSES if exposure[c]},
            "months_since_last_platinum": months(max(0, start - last_platinum)) if last_platinum is not None else None,
            "months_since_last_line": months(max(0, start - prev_end)) if prev_end is not None else None,
            "undated_history": static["undated_history"],
        }

        next_start = lines[index + 1]["start"] if index + 1 < len(lines) else None
        response_window_end = min(start + RESPONSE_MAX_DAYS, next_start) if next_start else start + RESPONSE_MAX_DAYS
        response = next((p["response"] for p in timeline
                         if start + RESPONSE_MIN_DAYS <= p["day"] <= response_window_end), None)
        # Progression proxy: explicit progression/recurrence day, the start of
        # the next line (time to next treatment), or death, whichever is first.
        # The cause is kept so the twin can split the event into progression or death.
        candidates = [(d, "progression") for d in progressions if d > start + PROGRESSION_MIN_DAYS]
        if next_start is not None:
            candidates.append((next_start, "next_line"))
        if death_day is not None and death_day >= start:
            candidates.append((death_day, "death"))
        first_event = min(candidates, key=lambda c: (c[0], c[1] == "death")) if candidates else None
        pfs_day = first_event[0] if first_event else None
        valid_death = death_day is not None and death_day >= start
        built.append({
            "line_index": index,
            "start_day": start,
            "end_day": line["end"],
            "regimen": line["classes"],
            "state": state,
            "outcome": {
                "response": 1 if response == "TF-Tumor Free" else 0 if response == "WT-With Tumor" else None,
                "pfs_months": months(pfs_day - start) if pfs_day is not None else months(max(0, last_day - start)),
                "pfs_event": int(pfs_day is not None),
                "pfs_event_is_death": int(first_event[1] == "death") if first_event else None,
                "os_months": months(death_day - start) if valid_death else months(max(0, last_day - start)),
                "os_event": int(valid_death),
                "next_setting": lines[index + 1]["setting"] if index + 1 < len(lines) else None,
            },
        })
    return {"patient_id": patient_id, "split": split, "static": static,
            "vital_status": clean(demographic.get("vital_status")) or None,
            "death_day_imputed": death_day_imputed,
            "undated_given_treatments": undated_given, "lines": built}


def build_histories(cases: list[dict], seed: int = 42) -> list[dict]:
    eligible = [c for c in cases if c.get("index_date") == "Diagnosis"]
    splits = assign_splits([str(c.get("case_id") or c.get("submitter_id")) for c in eligible], seed)
    return [build_patient(c, splits[str(c.get("case_id") or c.get("submitter_id"))]) for c in eligible]


def summarize(histories: list[dict]) -> dict:
    lines = [line for p in histories for line in p["lines"]]
    regimen_counts = Counter("+".join(line["regimen"]) for p in histories if p["split"] == "train" for line in p["lines"])
    return {
        "patients": len(histories),
        "patients_by_split": dict(Counter(p["split"] for p in histories)),
        "patients_with_lines": sum(bool(p["lines"]) for p in histories),
        "patients_with_undated_given_treatment": sum(p["undated_given_treatments"] > 0 for p in histories),
        "lines": len(lines),
        "lines_per_patient": dict(sorted(Counter(len(p["lines"]) for p in histories).items())),
        "line_settings": dict(Counter(line["state"]["setting"] for line in lines)),
        "response_labels": dict(Counter(str(line["outcome"]["response"]) for line in lines)),
        "pfs_events": sum(line["outcome"]["pfs_event"] for line in lines),
        "pfs_events_that_are_deaths": sum(line["outcome"]["pfs_event_is_death"] or 0 for line in lines),
        "os_events": sum(line["outcome"]["os_event"] for line in lines),
        "patients_dead_without_death_day": sum(p["death_day_imputed"] for p in histories),
        "train_regimens": dict(regimen_counts.most_common()),
        "definitions": {
            "line_window_days": LINE_WINDOW_DAYS,
            "response": f"first known follow-up response {RESPONSE_MIN_DAYS}-{RESPONSE_MAX_DAYS} days after line start and before the next line; 1 = TF-Tumor Free",
            "pfs": "months to the first of explicit progression/recurrence, next line start (time to next treatment), or death; otherwise censored at last contact. pfs_event_is_death records whether that first event was death",
            "os": "months from line start to death; otherwise censored at last contact. Patients recorded dead without a death day die at last contact",
            "undated_history": "patient-level flag kept for information; not a model input because undated records cannot be placed before or after a line",
            "excluded_records": "treatment_or_therapy = 'no', surgery, and records without a mappable class",
        },
    }


def write_histories(histories: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "patient_histories.jsonl").open("w", encoding="utf-8") as stream:
        for patient in histories:
            stream.write(json.dumps(patient) + "\n")
    rows = []
    for patient in histories:
        for line in patient["lines"]:
            state = {k: v for k, v in line["state"].items() if k != "prior_exposure"}
            rows.append({"patient_id": patient["patient_id"], "split": patient["split"],
                         "line_index": line["line_index"], "start_day": line["start_day"],
                         "end_day": line["end_day"], "regimen": "+".join(line["regimen"]),
                         "prior_exposure": json.dumps(line["state"]["prior_exposure"]),
                         **{f"state_{k}": v for k, v in state.items()},
                         **{f"outcome_{k}": v for k, v in line["outcome"].items()}})
    if rows:
        with (output_dir / "treatment_lines.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(histories)
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_histories(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
