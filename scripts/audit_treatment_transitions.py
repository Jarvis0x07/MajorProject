#!/usr/bin/env python3
"""Audit candidate TCGA-OV treatment-to-follow-up transition support.

Groups treatment records from the same patient on the same numeric start day as
one provisional same-day treatment set. This is an observational linkage audit,
not a causal treatment-effect estimator or an RL approval test.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


UNKNOWN = {"", "unknown", "not reported", "not available", "not applicable"}


def day(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def known(value):
    return value is not None and str(value).strip().lower() not in UNKNOWN


def lag_summary(episodes, after_key, action_day_key):
    lags = [e[after_key]["day"] - e[action_day_key] for e in episodes if e.get(after_key)]
    if not lags:
        return {"count": 0, "min_days": None, "median_days": None, "p90_days": None, "max_days": None}
    ordered = sorted(lags)
    return {
        "count": len(ordered), "min_days": ordered[0],
        "median_days": statistics.median(ordered),
        "p90_days": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "max_days": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TCGA-OV case JSON array")
    parser.add_argument("--output", default="data/processed/treatment_transition_audit.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Expected the top-level JSON value to be a list")

    total_treatments = 0
    treatments_with_start = 0
    treatments_with_end = 0
    treatments_negative_start = 0
    treatments_unknown_agent = 0
    treatments_unknown_type = 0
    treatments_unknown_dose = 0
    cases_with_untimed_treatments = 0
    eligible_cases_with_untimed_treatments = 0
    eligible_case_count = sum(c.get("index_date") == "Diagnosis" for c in cases)
    excluded_index_cases = len(cases) - eligible_case_count
    patient_results = {}
    patient_complete_start_timeline = {}
    patient_has_treatment_rows = {}
    action_type_sets = Counter()
    action_agent_strings = Counter()
    outcome_by_type = defaultdict(Counter)
    same_day_followup_conflicts = 0
    same_day_response_ambiguities = 0
    same_day_followup_duplicates = 0
    same_day_treatment_groups = 0
    group_size_counts = Counter()
    raw_same_day_response_ambiguities = 0

    for case_no, case in enumerate(cases, 1):
        treatment_rows = []
        case_untimed_treatments = 0
        case_treatment_record_count = 0
        for diagnosis_no, diagnosis in enumerate(case.get("diagnoses") or []):
            diagnosis_key = diagnosis.get("diagnosis_id") or diagnosis.get("submitter_id") or f"diagnosis_{diagnosis_no}"
            for treatment in diagnosis.get("treatments") or []:
                total_treatments += 1
                case_treatment_record_count += 1
                start = day(treatment.get("days_to_treatment_start"))
                end = day(treatment.get("days_to_treatment_end"))
                if start is not None:
                    treatments_with_start += 1
                    treatments_negative_start += start < 0
                else:
                    case_untimed_treatments += 1
                if end is not None:
                    treatments_with_end += 1
                agent = treatment.get("therapeutic_agents")
                ttype = treatment.get("treatment_type")
                dose = treatment.get("treatment_dose")
                if not known(agent):
                    treatments_unknown_agent += 1
                if not known(ttype):
                    treatments_unknown_type += 1
                if dose is None:
                    treatments_unknown_dose += 1
                if start is not None and start >= 0:
                    treatment_rows.append({
                        "start": start, "end": end,
                        "type": str(ttype).strip() if known(ttype) else "Unknown",
                        "agent": str(agent).strip() if known(agent) else "Unknown",
                        "dose": dose,
                        "dose_unit": treatment.get("treatment_dose_units"),
                        "diagnosis_key": diagnosis_key,
                    })

        if case_untimed_treatments:
            cases_with_untimed_treatments += 1
            if case.get("index_date") == "Diagnosis":
                eligible_cases_with_untimed_treatments += 1
        if case.get("index_date") == "Diagnosis":
            patient_complete_start_timeline[case_no] = case_untimed_treatments == 0
            patient_has_treatment_rows[case_no] = case_treatment_record_count > 0

        followup_by_day = defaultdict(set)
        followup_records_by_day = Counter()
        for followup in case.get("follow_ups") or []:
            d = day(followup.get("days_to_follow_up"))
            response = followup.get("disease_response")
            if d is None or not known(response):
                continue
            followup_by_day[d].add(str(response).strip())
            followup_records_by_day[d] += 1
        same_day_response_ambiguities += sum(len(v) > 1 for v in followup_by_day.values())
        same_day_followup_duplicates += sum(max(0, n - 1) for n in followup_records_by_day.values())
        response_timeline = [(d, next(iter(labels))) for d, labels in sorted(followup_by_day.items()) if len(labels) == 1]
        conflicting_days = {d for d, labels in followup_by_day.items() if len(labels) > 1}
        same_day_followup_conflicts += len(conflicting_days)

        if case.get("index_date") != "Diagnosis":
            continue

        by_start = defaultdict(list)
        for row in treatment_rows:
            by_start[row["start"]].append(row)
        episodes = []
        all_start_days = sorted(by_start)
        for start, members in sorted(by_start.items()):
            same_day_treatment_groups += len(members) > 1
            group_size_counts[str(len(members))] += 1
            type_set = tuple(sorted({m["type"] for m in members}))
            action_type_sets[" + ".join(type_set)] += 1
            for agent in {m["agent"] for m in members}:
                if agent != "Unknown":
                    action_agent_strings[agent] += 1

            after_start = next(((d, label) for d, label in response_timeline if d > start), None)
            intervening_starts_after_start = [d for d in all_start_days if start < d < after_start[0]] if after_start else []
            all_ends_known = all(m["end"] is not None and m["end"] >= start for m in members)
            last_end = max(m["end"] for m in members) if all_ends_known else None
            after_end = next(((d, label) for d, label in response_timeline if d > last_end), None) if last_end is not None else None
            intervening_starts_after_end = [d for d in all_start_days if last_end < d < after_end[0]] if after_end and last_end is not None else []
            if after_start:
                outcome_by_type["after_start"][" + ".join(type_set)] += 1
            if after_end:
                outcome_by_type["after_end"][" + ".join(type_set)] += 1

            # If a known response is on the action day, it is excluded from the
            # strict-after-start candidate; record that ambiguity separately.
            raw_same_day_response_ambiguities += start in followup_by_day
            episodes.append({
                "start_day": start,
                "member_count": len(members),
                "treatment_types": list(type_set),
                "agents": sorted({m["agent"] for m in members}),
                "all_member_end_days_observed": all_ends_known,
                "last_member_end_day": last_end,
                "first_known_response_after_start": ({"day": after_start[0], "label": after_start[1]} if after_start else None),
                "first_known_response_after_all_ends": ({"day": after_end[0], "label": after_end[1]} if after_end else None),
                "parent_diagnosis_count": len({m["diagnosis_key"] for m in members}),
                "intervening_treatment_start_days_before_first_response": intervening_starts_after_start,
                "intervening_treatment_start_days_after_completion_before_first_response": intervening_starts_after_end,
            })
        if episodes:
            patient_results[case_no] = episodes

    all_episodes = [e for groups in patient_results.values() for e in groups]
    fully_start_timed_episodes = [e for p, groups in patient_results.items()
                                  if patient_complete_start_timeline.get(p) for e in groups]
    fully_start_timed_completion_links = [e for e in fully_start_timed_episodes
                                          if e["first_known_response_after_all_ends"]]
    fully_start_timed_strict_links = [e for e in fully_start_timed_completion_links
                                      if not e["intervening_treatment_start_days_before_first_response"]
                                      and e["parent_diagnosis_count"] == 1]
    with_start_linked = [e for e in all_episodes if e["first_known_response_after_start"]]
    with_end_linked = [e for e in all_episodes if e["first_known_response_after_all_ends"]]
    isolated_start_linked = [e for e in with_start_linked if not e["intervening_treatment_start_days_before_first_response"]]
    no_post_completion_action = [e for e in with_end_linked if not e["intervening_treatment_start_days_after_completion_before_first_response"]]
    # For stronger attribution hygiene, also disallow treatment starts between
    # this action's start and the outcome, including overlapping treatment.
    isolated_completion_linked = [e for e in with_end_linked if not e["intervening_treatment_start_days_before_first_response"]]
    single_diagnosis_completion_linked = [e for e in isolated_completion_linked if e["parent_diagnosis_count"] == 1]
    end_observed_episodes = [e for e in all_episodes if e["all_member_end_days_observed"]]
    linked_patients_start = {p for p, es in patient_results.items() if any(e["first_known_response_after_start"] for e in es)}
    linked_patients_end = {p for p, es in patient_results.items() if any(e["first_known_response_after_all_ends"] for e in es)}
    patients_isolated_completion = {p for p, es in patient_results.items() if any(
        e["first_known_response_after_all_ends"] and not e["intervening_treatment_start_days_before_first_response"]
        and e["parent_diagnosis_count"] == 1 for e in es)}
    strict_action_counts = Counter(" + ".join(e["treatment_types"]) for e in single_diagnosis_completion_linked)

    # Examples are anonymized and deterministic: no case IDs or submitter IDs.
    examples = []
    for index, episode in enumerate(single_diagnosis_completion_linked[:5], 1):
        examples.append({"example": f"linked_{index}", **episode})
    for index, episode in enumerate([e for e in all_episodes if not e["first_known_response_after_start"]][:3], 1):
        examples.append({"example": f"no_later_response_{index}", **episode})

    report = {
        "study": "Exploratory same-day treatment-set to subsequent observed follow-up response linkage audit",
        "input_file": Path(args.input).name,
        "case_count": len(cases),
        "time_origin_policy": {
            "eligible_case_rule": "index_date == Diagnosis",
            "treatment_day_basis": "The TCGA pharmaceutical supplemental form defines therapy start/end day intervals from initial pathologic diagnosis; exact export-to-form mapping still requires verification for each included treatment class.",
            "follow_up_day_basis": "The TCGA Ovary Followup Form defines follow-up intervals from initial pathologic diagnosis.",
            "event_linkage_rule": "Same-case treatment records with the same numeric start day are provisionally bundled; the first unambiguous known disease_response at a strictly later follow-up day is counted.",
            "completion_linkage_rule": "Stricter sensitivity count requires every treatment record in the same-day bundle to have a numeric end day >= start; outcome must be at a strictly later day than the latest end day.",
        },
        "coverage": {
            "eligible_cases_index_date_diagnosis": eligible_case_count,
            "excluded_cases_missing_or_other_index_date": excluded_index_cases,
            "all_treatment_records": total_treatments,
            "treatment_records_with_numeric_start_day": treatments_with_start,
            "treatment_records_without_numeric_start_day": total_treatments - treatments_with_start,
            "cases_with_at_least_one_treatment_without_numeric_start_day": cases_with_untimed_treatments,
            "eligible_cases_with_at_least_one_treatment_without_numeric_start_day": eligible_cases_with_untimed_treatments,
            "treatment_records_with_numeric_end_day": treatments_with_end,
            "treatment_records_negative_start_day": treatments_negative_start,
            "negative_start_records_excluded_from_candidate_timeline": treatments_negative_start,
            "treatment_records_without_known_agent": treatments_unknown_agent,
            "treatment_records_without_known_type": treatments_unknown_type,
            "treatment_records_without_dose_value": treatments_unknown_dose,
            "known_response_followup_days_with_conflicting_labels": same_day_followup_conflicts,
            "duplicate_known_response_records_on_same_day_beyond_first": same_day_followup_duplicates,
        },
        "candidate_transition_counts": {
            "patients_with_at_least_one_timeable_treatment_set": len(patient_results),
            "eligible_cases_with_treatment_records_and_start_day_on_every_recorded_treatment": sum(
                patient_complete_start_timeline[p] and patient_has_treatment_rows[p] for p in patient_complete_start_timeline
            ),
            "same_day_treatment_sets": len(all_episodes),
            "treatment_sets_with_multiple_records": sum(e["member_count"] > 1 for e in all_episodes),
            "treatment_set_size_distribution": dict(sorted(group_size_counts.items(), key=lambda x: int(x[0]))),
            "sets_with_known_response_strictly_after_start": len(with_start_linked),
            "patients_with_known_response_after_at_least_one_start": len(linked_patients_start),
            "sets_with_response_after_start_and_no_intervening_treatment_start": len(isolated_start_linked),
            "sets_with_all_end_days_and_known_response_after_completion": len(with_end_linked),
            "patients_with_known_response_after_at_least_one_completion": len(linked_patients_end),
            "sets_with_all_member_end_days_observed": len(end_observed_episodes),
            "sets_with_response_after_completion_and_no_treatment_start_after_completion_before_response": len(no_post_completion_action),
            "sets_with_response_after_completion_and_no_treatment_start_between_action_start_and_response": len(isolated_completion_linked),
            "sets_with_response_after_completion_no_intervening_treatment_and_single_parent_diagnosis": len(single_diagnosis_completion_linked),
            "patients_with_at_least_one_strict_isolated_completion_link": len(patients_isolated_completion),
            "sets_with_all_treatment_starts_observed_for_patient_and_strict_post_completion_response": len(fully_start_timed_strict_links),
            "patients_with_all_treatment_starts_observed_and_at_least_one_strict_post_completion_response": len({
                p for p, groups in patient_results.items() if patient_complete_start_timeline.get(p)
                and any(e in fully_start_timed_strict_links for e in groups)
            }),
            "sets_without_any_later_unambiguous_known_response": len(all_episodes) - len(with_start_linked),
            "sets_with_same_day_known_response_ambiguous_for_order": raw_same_day_response_ambiguities,
        },
        "action_support": {
            "same_day_sets_by_treatment_type_combination_top_25": [
                {"type_combination": k, "sets": n} for k, n in action_type_sets.most_common(25)
            ],
            "therapeutic_agent_strings_top_25": [
                {"agent_string": k, "sets_containing_agent": n} for k, n in action_agent_strings.most_common(25)
            ],
            "post_start_links_by_action_type_top_25": [
                {"type_combination": k, "sets": n} for k, n in outcome_by_type["after_start"].most_common(25)
            ],
            "post_completion_links_by_action_type_top_25": [
                {"type_combination": k, "sets": n} for k, n in outcome_by_type["after_end"].most_common(25)
            ],
            "strict_isolated_post_completion_sets_by_action_type": [
                {"type_combination": k, "sets": n}
                for k, n in strict_action_counts.most_common()
            ],
        },
        "response_lag_days": {
            "from_treatment_start_to_first_later_known_response": lag_summary(with_start_linked, "first_known_response_after_start", "start_day"),
            "from_latest_treatment_end_to_first_later_known_response": {
                "count": len(with_end_linked),
                "min_days": min(e["first_known_response_after_all_ends"]["day"] - e["last_member_end_day"] for e in with_end_linked) if with_end_linked else None,
                "median_days": statistics.median([e["first_known_response_after_all_ends"]["day"] - e["last_member_end_day"] for e in with_end_linked]) if with_end_linked else None,
                "max_days": max(e["first_known_response_after_all_ends"]["day"] - e["last_member_end_day"] for e in with_end_linked) if with_end_linked else None,
            },
            "strict_isolated_post_completion_links": lag_summary(single_diagnosis_completion_linked, "first_known_response_after_all_ends", "last_member_end_day"),
            "strict_isolated_post_completion_response_labels": dict(Counter(e["first_known_response_after_all_ends"]["label"] for e in single_diagnosis_completion_linked)),
        },
        "anonymized_examples": examples,
        "interpretation_and_limits": [
            "Same-day grouping is a provisional data aggregation, not proof that grouped entries form a prescribed clinical regimen.",
            "A later follow-up response is an observed association after treatment start/completion; it is not attributable to that treatment and is not a counterfactual treatment effect.",
            "Isolation counts only exclude intervening treatment starts that have recorded numeric dates. Undated treatment records mean true isolation cannot be confirmed; strict counts are upper-bound candidate links.",
            "Follow-up response is sparse, irregular, and may represent a status at last contact or a named timepoint rather than a fixed-horizon outcome.",
            "Dose, agent, end date, side effects, and other clinically important fields are often missing; an action label may therefore be incomplete.",
            "These counts assess feasibility and do not approve this cohort, schema, simulator, or policy for RL.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved audit to {out.resolve()}")


if __name__ == "__main__":
    main()
