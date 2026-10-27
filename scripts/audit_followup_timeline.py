#!/usr/bin/env python3
"""Audit TCGA-OV follow-up time fields and response ordering."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


UNKNOWN = {"unknown", "not reported", "not available", ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/followup_timeline_audit.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))

    index_dates = Counter(str(c.get("index_date")) for c in cases if c.get("index_date") is not None)
    day_values = []
    response_records_timed = 0
    response_records_without_day = 0
    nonnumeric_response_days = 0
    source_order_nonmonotonic_cases = 0
    patients_with_2plus_response_records = 0
    patients_with_2plus_distinct_response_days = 0
    response_transitions = 0
    response_labels = Counter()
    same_day_conflicts = 0
    examples = []

    for case in cases:
        followups = case.get("follow_ups") or []
        for followup in followups:
            day = followup.get("days_to_follow_up")
            if day is not None:
                try:
                    day_values.append(int(day))
                except (ValueError, TypeError):
                    pass

        timed = []
        by_day = defaultdict(set)
        for followup in followups:
            response = followup.get("disease_response")
            if response is None or str(response).strip().lower() in UNKNOWN:
                continue
            day = followup.get("days_to_follow_up")
            if day is None:
                response_records_without_day += 1
                continue
            try:
                day = int(day)
            except (ValueError, TypeError):
                nonnumeric_response_days += 1
                continue
            response_records_timed += 1
            response = str(response).strip()
            response_labels[response] += 1
            timed.append(day)
            by_day[day].add(response)

        if len(timed) >= 2:
            patients_with_2plus_response_records += 1
            if timed != sorted(timed):
                source_order_nonmonotonic_cases += 1
        conflicts = [day for day, labels in by_day.items() if len(labels) > 1]
        same_day_conflicts += len(conflicts)
        timeline = [(day, next(iter(labels))) for day, labels in sorted(by_day.items()) if len(labels) == 1]
        if len(timeline) >= 2:
            patients_with_2plus_distinct_response_days += 1
            response_transitions += len(timeline) - 1
            if len(examples) < 3:
                examples.append({
                    "case_label": f"example_{len(examples) + 1}",
                    "timed_response_records_in_source_order": [
                        {"day": int(f["days_to_follow_up"]), "response": str(f["disease_response"]).strip()}
                        for f in followups if f.get("days_to_follow_up") is not None
                        and f.get("disease_response") not in (None, "")
                        and str(f.get("disease_response")).strip().lower() not in UNKNOWN
                        and str(f.get("days_to_follow_up")).lstrip("-").isdigit()],
                    "ordered_distinct_timepoints": [
                        {"day": day, "response": response} for day, response in timeline
                    ],
                })

    day_count = len(day_values)
    audit = {
        "dataset_case_count": len(cases),
        "index_date_values": dict(index_dates),
        "follow_up_day_records": day_count,
        "follow_up_day_type_counts": {"integer_values": sum(isinstance(f.get("days_to_follow_up"), int) for c in cases for f in (c.get("follow_ups") or []) if f.get("days_to_follow_up") is not None),
                                      "other_numeric_or_string_values": day_count - sum(isinstance(f.get("days_to_follow_up"), int) for c in cases for f in (c.get("follow_ups") or []) if f.get("days_to_follow_up") is not None)},
        "follow_up_day_range": [min(day_values), max(day_values)] if day_values else None,
        "negative_follow_up_days": sum(day < 0 for day in day_values),
        "known_response_records_with_numeric_day": response_records_timed,
        "known_response_records_missing_day": response_records_without_day,
        "known_response_records_nonnumeric_day": nonnumeric_response_days,
        "known_response_labels": dict(response_labels),
        "patients_with_two_or_more_timed_known_response_records": patients_with_2plus_response_records,
        "those_patients_with_nonchronological_source_array_order": source_order_nonmonotonic_cases,
        "same_day_conflicting_response_groups": same_day_conflicts,
        "patients_with_two_or_more_distinct_unambiguous_response_timepoints": patients_with_2plus_distinct_response_days,
        "adjacent_distinct_response_transitions": response_transitions,
        "manual_examples": examples,
        "interpretation": [
            "TCGA follow-up intervals are documented as measured from initial pathologic diagnosis; GDC uses the study's index date. This export's index_date values must be checked alongside that source convention.",
            "Follow-up arrays are not always in chronological order. The evaluator sorts by numeric days_to_follow_up and collapses same-day duplicate labels.",
            "Distinct numeric dates establish ordering only; they do not establish regular intervals or an observed clinical assessment on every day.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\nSaved audit to {out.resolve()}")


if __name__ == "__main__":
    main()
