#!/usr/bin/env python3
"""Legacy extractor: flatten a TCGA-OV export into a traceable event table.

This script is an extraction prototype, not a validated clinical timeline builder.
It preserves the original relative-day values and does NOT assume that every
`days_to_*` field shares the same time origin. Review the source data dictionary
before using `event_day` to construct model transitions.

Usage:
  python scripts/preprocess_tcga_ov.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --output-dir data/processed

The source JSON is not modified. Output files:
  patient_events.csv
  data_quality_summary.json

For the base-paper-inspired 11-class action grouping and candidate state/action/
outcome table, use `scripts/paper_method_pipeline.py`. This extractor does not
construct an RL dataset or guarantee comparable event-time origins.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def first_present(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else ""


def add_event(rows, case, event_type, record, source_path, day_keys, parent_diagnosis_id=None):
    day_value = first_present(record, day_keys)
    # Preserve the source value. It is not asserted to be a globally comparable day.
    rows.append({
        "case_id": case.get("case_id", ""),
        "patient_submitter_id": case.get("submitter_id", ""),
        "event_type": event_type,
        "event_day_raw": day_value if day_value is not None else "",
        "event_day_field": next((k for k in day_keys if record.get(k) not in (None, "")), ""),
        "source_path": source_path,
        "source_record_id": first_present(record, ["diagnosis_id", "treatment_id", "follow_up_id", "pathology_detail_id", "submitter_id"]) or "",
        "parent_diagnosis_id": parent_diagnosis_id or "",
        "event_payload_json": safe_json(record),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the source TCGA-OV JSON file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("Expected the top-level JSON value to be a list of case records")

    rows = []
    quality = {
        "input_file_name": input_path.name,
        "case_count": len(cases),
        "event_counts": {},
        "cases_without_demographic_object": 0,
        "cases_without_diagnoses_array": 0,
        "cases_without_follow_ups_array": 0,
        "treatments_without_start_day": 0,
        "follow_ups_without_followup_day": 0,
        "warning": "event_day_raw preserves the original field value; different days_to_* fields may have different reference points. Verify before sorting into a clinical timeline.",
    }
    counts = Counter()

    for case in cases:
        cid = case.get("case_id", "")
        if not case.get("demographic"):
            quality["cases_without_demographic_object"] += 1
        if not isinstance(case.get("diagnoses"), list):
            quality["cases_without_diagnoses_array"] += 1
        if not isinstance(case.get("follow_ups"), list):
            quality["cases_without_follow_ups_array"] += 1

        # Patient baseline is an event at the index/diagnosis reference, but has no
        # fabricated day number. Static fields are carried in the payload.
        baseline = {
            "index_date": case.get("index_date"),
            "disease_type": case.get("disease_type"),
            "primary_site": case.get("primary_site"),
            "demographic": case.get("demographic"),
        }
        add_event(rows, case, "patient_baseline", baseline, "case", [], None)
        counts["patient_baseline"] += 1

        for di, diagnosis in enumerate(case.get("diagnoses") or []):
            did = diagnosis.get("diagnosis_id") or f"diagnosis_index_{di}"
            add_event(rows, case, "diagnosis", diagnosis, f"diagnoses[{di}]", ["days_to_diagnosis"], did)
            counts["diagnosis"] += 1

            for ti, treatment in enumerate(diagnosis.get("treatments") or []):
                if first_present(treatment, ["days_to_treatment_start"]) is None:
                    quality["treatments_without_start_day"] += 1
                add_event(rows, case, "treatment", treatment,
                          f"diagnoses[{di}].treatments[{ti}]",
                          ["days_to_treatment_start"], did)
                counts["treatment"] += 1

            for pi, pathology in enumerate(diagnosis.get("pathology_details") or []):
                add_event(rows, case, "pathology", pathology,
                          f"diagnoses[{di}].pathology_details[{pi}]", [], did)
                counts["pathology"] += 1

        for fi, follow_up in enumerate(case.get("follow_ups") or []):
            if first_present(follow_up, ["days_to_follow_up"]) is None:
                quality["follow_ups_without_followup_day"] += 1
            add_event(rows, case, "follow_up", follow_up, f"follow_ups[{fi}]",
                      ["days_to_follow_up"], None)
            counts["follow_up"] += 1

    # Stable grouping by case and source sequence. Do not sort on raw days because
    # their reference points have not yet been verified.
    out_csv = output_dir / "patient_events.csv"
    fieldnames = ["case_id", "patient_submitter_id", "event_type", "event_day_raw",
                  "event_day_field", "source_path", "source_record_id",
                  "parent_diagnosis_id", "event_payload_json"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    quality["event_counts"] = dict(counts)
    quality["event_row_count"] = len(rows)
    quality["output_csv"] = str(out_csv)
    with (output_dir / "data_quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(cases)} cases into {len(rows)} event rows.")
    print(f"Events: {dict(counts)}")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("IMPORTANT: verify time origins before constructing a chronological state-transition dataset.")


if __name__ == "__main__":
    main()
