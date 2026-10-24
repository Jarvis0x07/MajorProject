#!/usr/bin/env python3
"""Leakage-aware patient-state construction and LLM trajectory interface.

This is a research prototype. It does not estimate causal treatment effects or
provide clinical recommendations. Relative-day fields are retained separately
because the TCGA export's time origins have not been verified.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _field(record: dict[str, Any], name: str) -> dict[str, Any]:
    """Keep absent, explicit missing-like labels, and observed values distinct."""
    if name not in record or record[name] is None or record[name] == "":
        return {"value": None, "status": "absent"}
    value = record[name]
    if isinstance(value, str) and value.strip().lower() in {"unknown", "not reported", "not available"}:
        return {"value": value, "status": "explicit_unknown"}
    return {"value": value, "status": "observed"}


def build_patient_state(case: dict[str, Any], as_of: int | None = None) -> dict[str, Any]:
    """Build a compact state. With as_of, include only events on a verified day axis.

    `as_of` is deliberately rejected for the mixed TCGA event fields: each
    days_to_* field has a potentially different origin. Use `None` for a
    provenance-preserving case summary, or provide verified time-axis metadata
    in a future adapter rather than silently ordering incomparable dates.
    """
    if as_of is not None:
        raise ValueError("as_of filtering is disabled until TCGA relative-day origins are verified")
    demographic = case.get("demographic") or {}
    diagnoses = case.get("diagnoses") or []
    followups = case.get("follow_ups") or []
    treatments = []
    diagnosis_history = []
    pathology = []
    for diagnosis in diagnoses:
        diagnosis_history.append({
            "classification": _field(diagnosis, "classification_of_tumor"),
            "primary_diagnosis": _field(diagnosis, "primary_diagnosis"),
            "figo_stage": _field(diagnosis, "figo_stage"),
            "tumor_grade": _field(diagnosis, "tumor_grade"),
            "days_to_diagnosis_raw": _field(diagnosis, "days_to_diagnosis"),
        })
        for treatment in diagnosis.get("treatments") or []:
            treatments.append({
                "treatment_type": _field(treatment, "treatment_type"),
                "therapeutic_agents": _field(treatment, "therapeutic_agents"),
                "dose": _field(treatment, "treatment_dose"),
                "dose_units": _field(treatment, "treatment_dose_units"),
                "cycles": _field(treatment, "number_of_cycles"),
                "start_day_raw": _field(treatment, "days_to_treatment_start"),
                "end_day_raw": _field(treatment, "days_to_treatment_end"),
                "recorded_outcome": _field(treatment, "treatment_outcome"),
            })
        for detail in diagnosis.get("pathology_details") or []:
            pathology.append({k: _field(detail, k) for k in (
                "vascular_invasion_present", "lymphatic_invasion_present", "residual_tumor_measurement")})

    # Outcomes are kept in a separate section; an LLM must not treat the full
    # retrospective record as information known at an earlier decision point.
    observed_outcomes = [{
        "timepoint": _field(f, "timepoint_category"),
        "response": _field(f, "disease_response"),
        "follow_up_day_raw": _field(f, "days_to_follow_up"),
        "progression": _field(f, "progression_or_recurrence"),
    } for f in followups]
    return {
        "schema_version": "tcga_ov_state_v1",
        "case_key": case.get("case_id") or case.get("submitter_id"),
        "time_axis_status": "unverified_relative_day_origins",
        "profile": {
            "age_at_index": _field(demographic, "age_at_index"),
            "sex_at_birth": _field(demographic, "sex_at_birth"),
            "race": _field(demographic, "race"),
            "ethnicity": _field(demographic, "ethnicity"),
            "primary_site": _field(case, "primary_site"),
        },
        "diagnosis_history": diagnosis_history,
        "treatment_history": treatments,
        "pathology": pathology,
        "retrospective_outcomes_not_safe_as_current_state": observed_outcomes,
        "state_construction_warning": "This is a retrospective record summary, not a point-in-time state. Exclude all later events/outcomes after clinical time origins and availability are verified.",
    }


def state_to_prompt(state: dict[str, Any], horizon: str = "next recorded follow-up") -> str:
    """Create a grounded, structured prompt for an approved LLM endpoint."""
    return "\n".join([
        "Task: produce a research-only forecast of the next observed disease-response category.",
        "Use only the supplied structured data. Do not recommend treatment or invent measurements.",
        "The record is retrospective and its relative-day origins are unverified; do not infer chronology across event types.",
        f"Forecast horizon: {horizon}.",
        "Return one JSON object with keys: prediction, confidence (0..1 or null), evidence_fields (array), limitations (array).",
        "If the data cannot support a forecast, set prediction to null and explain why in limitations.",
        "Patient record JSON:", json.dumps(state, ensure_ascii=False, sort_keys=True),
    ])


@dataclass
class TrajectoryForecast:
    prediction: str | None
    confidence: float | None
    evidence_fields: list[str]
    limitations: list[str]


def predict_next_trajectory(state: dict[str, Any], model: Callable[[str], str], horizon: str = "next recorded follow-up") -> TrajectoryForecast:
    """Invoke an injected LLM; validate its JSON response and uncertainty."""
    raw = model(state_to_prompt(state, horizon))
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("LLM response must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    prediction = data.get("prediction")
    confidence = data.get("confidence")
    if prediction is not None and not isinstance(prediction, str):
        raise ValueError("prediction must be a string or null")
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1):
        raise ValueError("confidence must be between 0 and 1, or null")
    for key in ("evidence_fields", "limitations"):
        if not isinstance(data.get(key), list) or not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"{key} must be an array of strings")
    # No output can make an unverified retrospective record a valid prospective state.
    limits = list(data["limitations"])
    if state.get("time_axis_status") == "unverified_relative_day_origins":
        limits.append("Retrospective summary; event time origins and point-in-time availability are unverified.")
    return TrajectoryForecast(prediction, float(confidence) if confidence is not None else None,
                              data["evidence_fields"], limits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TCGA-OV case JSON array")
    parser.add_argument("--output", required=True, help="Output JSONL state file")
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Expected a top-level JSON array of cases")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(build_patient_state(case), ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} patient summaries to {out.resolve()}")
    print("Warning: these are retrospective summaries, not validated point-in-time states.")


if __name__ == "__main__":
    main()
