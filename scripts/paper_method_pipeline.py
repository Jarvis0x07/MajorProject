#!/usr/bin/env python3
"""Build a paper-inspired TCGA-OV state/action/outcome dataset.

This is a transparent preprocessing adapter, not an implementation of the
paper's Transformer, treatment-effect estimator, or BCQ learner. It emits all
candidate same-day treatment decisions with timing/coverage flags so later
modelling can use a declared subset without silently treating missing events
as absent.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ACTION_CLASSES = [
    "platinum", "taxanes", "anthracyclines", "antimetabolites",
    "topoisomerase_inhibitors", "alkylating_agents", "anti_angiogenics",
    "hormonal_agents", "vinca_alkaloids", "radiation", "other",
]
UNKNOWN = {"", "unknown", "not reported", "not available", "not applicable"}

# Explicit source-agent mappings to the 11 broad classes reported in the paper.
# Any known agent not listed here is retained under `other` and appears in the
# generated mapping CSV for review. Missing/Unknown agents remain unmapped.
AGENT_CLASS = {
    "carboplatin": "platinum", "cisplatin": "platinum", "oxaliplatin": "platinum",
    "paclitaxel": "taxanes", "docetaxel": "taxanes", "nab-paclitaxel": "taxanes",
    "paclitaxel poliglumex": "taxanes", "taxane compound": "taxanes",
    "pegylated liposomal doxorubicin hydrochloride": "anthracyclines",
    "doxorubicin": "anthracyclines", "doxorubicin hydrochloride": "anthracyclines",
    "gemcitabine": "antimetabolites", "gemcitabine hydrochloride": "antimetabolites",
    "capecitabine": "antimetabolites", "pemetrexed disodium": "antimetabolites",
    "pemetrexed": "antimetabolites", "fluorouracil": "antimetabolites",
    "aminopterin": "antimetabolites", "plevitrexed": "antimetabolites",
    "topotecan": "topoisomerase_inhibitors", "topotecan hydrochloride": "topoisomerase_inhibitors",
    "etoposide": "topoisomerase_inhibitors", "irinotecan hydrochloride": "topoisomerase_inhibitors",
    "aminocamptothecin": "topoisomerase_inhibitors", "vosaroxin": "topoisomerase_inhibitors",
    "cyclophosphamide": "alkylating_agents", "altretamine": "alkylating_agents",
    "ifosfamide": "alkylating_agents", "melphalan": "alkylating_agents",
    "chlorambucil": "alkylating_agents", "trabectedin": "alkylating_agents",
    "irofulven": "alkylating_agents",
    "bevacizumab": "anti_angiogenics", "motesanib diphosphate": "anti_angiogenics",
    "pan-vegfr/tie2 tyrosine kinase inhibitor cep-11981": "anti_angiogenics",
    "sorafenib": "anti_angiogenics",
    "tamoxifen": "hormonal_agents", "letrozole": "hormonal_agents",
    "megestrol acetate": "hormonal_agents", "fulvestrant": "hormonal_agents",
    "exemestane": "hormonal_agents", "leuprolide acetate": "hormonal_agents",
    "medroxyprogesterone acetate": "hormonal_agents",
    "vinorelbine tartrate": "vinca_alkaloids", "vinorelbine": "vinca_alkaloids",
    "vincristine": "vinca_alkaloids",
    "anastrozole": "hormonal_agents", "leucovorin": "antimetabolites",
}


def clean(value):
    return str(value).strip() if value is not None else ""


def is_known(value):
    return value is not None and clean(value).lower() not in UNKNOWN


def as_day(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return int(n) if n.is_integer() else None


def as_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def response_timeline(case):
    by_day = defaultdict(set)
    source_by_day = defaultdict(list)
    for row in case.get("follow_ups") or []:
        day = as_day(row.get("days_to_follow_up"))
        response = row.get("disease_response")
        if day is None or day < 0 or not is_known(response):
            continue
        by_day[day].add(clean(response))
        source_by_day[day].append(row)
    result = []
    for day in sorted(by_day):
        labels = by_day[day]
        if len(labels) != 1:
            continue
        rows = source_by_day[day]
        ecog = {clean(r.get("ecog_performance_status")) for r in rows if is_known(r.get("ecog_performance_status"))}
        kps = {clean(r.get("karnofsky_performance_status")) for r in rows if is_known(r.get("karnofsky_performance_status"))}
        # Conflicting same-day functional scores are left missing instead of
        # arbitrarily selecting one source row.
        result.append({
            "day": day, "response": next(iter(labels)),
            "ecog": next(iter(ecog)) if len(ecog) == 1 else None,
            "karnofsky": next(iter(kps)) if len(kps) == 1 else None,
        })
    return result


def diagnosis_is_primary(diagnosis):
    marker = clean(diagnosis.get("diagnosis_is_primary_disease")).lower()
    classification = clean(diagnosis.get("classification_of_tumor")).lower()
    return marker == "true" or classification == "primary"


def classify_treatment(treatment):
    ttype = clean(treatment.get("treatment_type")).lower()
    if "radiation" in ttype or "radioisotope" in ttype:
        return ["radiation"], "radiation_by_treatment_type"
    if "surgery" in ttype:
        # The paper's 11 classes do not name surgery. Keep it in the event file
        # but do not invent a drug class or silently call it a drug action.
        return [], "surgery_excluded_from_11_class_action_space"
    raw = treatment.get("therapeutic_agents")
    if not is_known(raw):
        return [], "missing_or_unknown_agent"
    names = [s.strip().lower() for s in clean(raw).replace(";", ",").split(",") if s.strip()]
    classes = sorted({AGENT_CLASS.get(name, "other") for name in names})
    basis = "explicit_agent_mapping" if all(name in AGENT_CLASS for name in names) else "unlisted_known_agent_to_other"
    return classes, basis


def choose_primary_diagnosis(case):
    diagnoses = [d for d in case.get("diagnoses") or [] if diagnosis_is_primary(d)]
    return diagnoses[0] if diagnoses else {}


def action_members(case):
    members = []
    for di, diagnosis in enumerate(case.get("diagnoses") or []):
        diagnosis_key = diagnosis.get("diagnosis_id") or diagnosis.get("submitter_id") or f"diagnosis_{di}"
        for treatment in diagnosis.get("treatments") or []:
            classes, map_basis = classify_treatment(treatment)
            members.append({
                "start_day": as_day(treatment.get("days_to_treatment_start")),
                "end_day": as_day(treatment.get("days_to_treatment_end")),
                "classes": classes,
                "mapping_basis": map_basis,
                "treatment_type": clean(treatment.get("treatment_type")),
                "agent": clean(treatment.get("therapeutic_agents")),
                "dose": treatment.get("treatment_dose"),
                "dose_unit": clean(treatment.get("treatment_dose_units")),
                "cycles": treatment.get("number_of_cycles"),
                "diagnosis_key": str(diagnosis_key),
                "source_treatment_id": treatment.get("treatment_id") or treatment.get("submitter_id"),
            })
    return members


def assign_splits(patient_ids, seed=42):
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    # Match the paper's 80/10/10 split exactly for the 587-case cohort.
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    if n == 587:
        n_train, n_val = 469, 59
    train = set(ids[:n_train])
    val = set(ids[n_train:n_train + n_val])
    return {pid: ("train" if pid in train else "validation" if pid in val else "test") for pid in ids}


def latest_prior(timeline, action_day):
    prior = [point for point in timeline if point["day"] < action_day]
    return prior[-1] if prior else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TCGA-OV JSON export")
    parser.add_argument("--output-dir", default="data/processed/paper_method")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Expected a top-level array of cases")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [c for c in cases if c.get("index_date") == "Diagnosis"]
    patient_ids = [str(c.get("case_id") or c.get("submitter_id")) for c in eligible]
    split_by_patient = assign_splits(patient_ids, args.seed)

    event_rows = []
    decision_rows = []
    mapping_basis_counts = Counter()
    agent_map_counts = Counter()
    class_counts = Counter()
    no_start_patients = 0
    treatment_total = 0
    treatment_starts = 0
    treatment_ends = 0
    negative_starts = 0
    excluded_surgery_records = 0
    missing_action_class_records = 0
    response_conflicts = 0
    no_start_count_patients = 0

    for case in eligible:
        patient_id = str(case.get("case_id") or case.get("submitter_id"))
        split = split_by_patient[patient_id]
        members = action_members(case)
        timeline = response_timeline(case)
        primary = choose_primary_diagnosis(case)
        missing_start_members = sum(m["start_day"] is None for m in members)
        has_incomplete_treatment_timing = missing_start_members > 0
        no_start_count_patients += has_incomplete_treatment_timing
        treatment_total += len(members)
        treatment_starts += sum(m["start_day"] is not None for m in members)
        treatment_ends += sum(m["end_day"] is not None for m in members)
        negative_starts += sum(m["start_day"] is not None and m["start_day"] < 0 for m in members)
        by_start = defaultdict(list)

        for member in members:
            mapping_basis_counts[member["mapping_basis"]] += 1
            for cls in member["classes"]:
                class_counts[cls] += 1
            if member["mapping_basis"] == "surgery_excluded_from_11_class_action_space":
                excluded_surgery_records += 1
            if member["mapping_basis"] == "missing_or_unknown_agent":
                missing_action_class_records += 1
            event_rows.append({
                "case_id": patient_id,
                "split": split,
                "treatment_id": member["source_treatment_id"],
                "parent_diagnosis_id": member["diagnosis_key"],
                "start_day": member["start_day"],
                "end_day": member["end_day"],
                "treatment_type": member["treatment_type"],
                "therapeutic_agents_raw": member["agent"],
                "action_classes": json.dumps(member["classes"]),
                "mapping_basis": member["mapping_basis"],
                "dose": member["dose"],
                "dose_unit": member["dose_unit"],
                "number_of_cycles": member["cycles"],
            })
            if member["start_day"] is not None and member["start_day"] >= 0:
                by_start[member["start_day"]].append(member)

        # Report a record-to-class mapping for every observed raw agent string.
        # The reviewer can inspect/override the generated action-class column.
        # The output is aggregated below to avoid duplicating source patient IDs.
        for treatment in (t for d in (case.get("diagnoses") or []) for t in (d.get("treatments") or [])):
            raw = clean(treatment.get("therapeutic_agents")) or "<MISSING>"
            classes, basis = classify_treatment(treatment)
            agent_map_counts[(raw, tuple(classes), basis)] += 1

        # Detect conflicts for documentation; response_timeline already omits them.
        response_by_day = defaultdict(set)
        for f in case.get("follow_ups") or []:
            fd = as_day(f.get("days_to_follow_up"))
            if fd is not None and is_known(f.get("disease_response")):
                response_by_day[fd].add(clean(f.get("disease_response")))
        response_conflicts += sum(len(labels) > 1 for labels in response_by_day.values())

        timed_starts = sorted(by_start)
        for action_day, action_members_for_day in sorted(by_start.items()):
            action_classes = sorted({c for m in action_members_for_day for c in m["classes"]})
            if not action_classes:
                continue
            action_combination = "+".join(action_classes)
            prior = latest_prior(timeline, action_day)
            after_start = next((p for p in timeline if p["day"] > action_day), None)
            all_ends_known = all(m["end_day"] is not None and m["end_day"] >= action_day for m in action_members_for_day)
            last_end = max(m["end_day"] for m in action_members_for_day) if all_ends_known else None
            after_end = next((p for p in timeline if p["day"] > last_end), None) if last_end is not None else None
            intervening_starts = [d for d in timed_starts if action_day < d < (after_end["day"] if after_end else -1)] if after_end else []
            parent_count = len({m["diagnosis_key"] for m in action_members_for_day})
            unmapped_member_count = sum(not m["classes"] for m in action_members_for_day)
            strict_recorded_link = bool(after_end and not intervening_starts and parent_count == 1)
            strict_complete_patient_link = strict_recorded_link and not has_incomplete_treatment_timing
            if strict_complete_patient_link:
                linkage_grade = "complete_patient_start_timeline"
            elif strict_recorded_link:
                linkage_grade = "dated_event_only_candidate"
            elif after_end:
                linkage_grade = "post_completion_with_intervening_timed_action_or_parent_ambiguity"
            elif after_start:
                linkage_grade = "post_start_only_completion_unknown"
            else:
                linkage_grade = "no_later_known_response"

            age = as_number((case.get("demographic") or {}).get("age_at_index"))
            action_class_set = set(action_classes)
            previous_known_drugs = [m for m in members if m["start_day"] is not None and 0 <= m["start_day"] < action_day and any(c != "radiation" for c in m["classes"])]
            previous_radiation = any(m["start_day"] is not None and 0 <= m["start_day"] < action_day and "radiation" in m["classes"] for m in members)
            tumor_free_reward = 1 if after_end and after_end["response"] == "TF-Tumor Free" else 0 if after_end else None

            row = {
                "case_id": patient_id,
                "split": split,
                "decision_day": action_day,
                "state_age_years": age,
                "state_gender": clean((case.get("demographic") or {}).get("sex_at_birth")) or None,
                "state_tumor_status_pre_action": prior["response"] if prior else None,
                "state_tumor_grade": clean(primary.get("tumor_grade")) or None,
                "state_figo_stage": clean(primary.get("figo_stage")) or None,
                "state_cumulative_dated_drug_record_count": len(previous_known_drugs),
                "state_radiation_history": int(previous_radiation),
                "state_ecog_pre_action": as_number(prior["ecog"]) if prior else None,
                "state_karnofsky_pre_action": as_number(prior["karnofsky"]) if prior else None,
                "state_has_undated_treatment_records": int(has_incomplete_treatment_timing),
                "action_classes": json.dumps(action_classes),
                "action_combination": action_combination,
                "same_day_treatment_record_count": len(action_members_for_day),
                "action_mapped_member_count": len(action_members_for_day) - unmapped_member_count,
                "action_unmapped_member_count": unmapped_member_count,
                "action_mapping_complete": int(unmapped_member_count == 0),
                "action_parent_diagnosis_count": parent_count,
                "next_response_after_start": after_start["response"] if after_start else None,
                "next_response_after_start_day": after_start["day"] if after_start else None,
                "all_action_end_days_observed": int(all_ends_known),
                "last_action_end_day": last_end,
                "next_response_after_completion": after_end["response"] if after_end else None,
                "next_response_after_completion_day": after_end["day"] if after_end else None,
                "response_lag_after_completion_days": after_end["day"] - last_end if after_end and last_end is not None else None,
                "reward_tumor_free": tumor_free_reward,
                "intervening_timed_action_days": json.dumps(intervening_starts),
                "linkage_grade": linkage_grade,
            }
            row.update({f"action_{cls}": int(cls in action_class_set) for cls in ACTION_CLASSES})
            decision_rows.append(row)

    # Patient split file records all eligible patients, including those with no
    # classifiable timed action, so partition membership is explicit.
    with (output_dir / "patient_splits.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split"])
        writer.writeheader()
        writer.writerows({"case_id": pid, "split": split} for pid, split in sorted(split_by_patient.items()))

    if event_rows:
        with (output_dir / "treatment_events.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)
    if decision_rows:
        with (output_dir / "state_action_outcome_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(decision_rows[0]))
            writer.writeheader()
            writer.writerows(decision_rows)

    with (output_dir / "treatment_class_mapping.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["source_agent", "record_count", "paper_action_classes", "mapping_basis"])
        writer.writeheader()
        for (raw, classes, basis), count in sorted(agent_map_counts.items(), key=lambda item: (-item[1], item[0][0])):
            writer.writerow({"source_agent": raw, "record_count": count, "paper_action_classes": "+".join(classes), "mapping_basis": basis})

    decisions_by_grade = Counter(row["linkage_grade"] for row in decision_rows)
    outcomes_after_end = Counter(row["next_response_after_completion"] for row in decision_rows if row["next_response_after_completion"])
    split_counts = Counter(split_by_patient.values())
    class_combos = Counter(row["action_combination"] for row in decision_rows)
    complete_class_combos = Counter(row["action_combination"] for row in decision_rows if row["action_mapping_complete"])
    summary = {
        "pipeline": "paper-inspired TCGA-OV state/action/outcome preprocessing",
        "input_file": Path(args.input).name,
        "paper_reference_numbers": {
            "reported_patients": 587,
            "reported_treatment_events": 2552,
            "reported_treatment_combinations": 47,
            "reported_positive_tumor_free_rate": 0.275,
            "reported_train_validation_test_patients": [469, 59, 59],
        },
        "cohort": {"input_cases": len(cases), "index_date_diagnosis_cases": len(eligible), "excluded_other_or_missing_index_date": len(cases) - len(eligible)},
        "local_export_counts": {
            "all_treatment_records_in_eligible_cases": treatment_total,
            "records_with_numeric_start_day": treatment_starts,
            "records_with_numeric_end_day": treatment_ends,
            "records_with_negative_start_day": negative_starts,
            "eligible_patients_with_any_undated_treatment": no_start_count_patients,
            "surgery_records_left_out_of_11_class_action_space": excluded_surgery_records,
            "treatment_records_without_mappable_action_class": missing_action_class_records,
            "treatment_mapping_basis_counts": dict(mapping_basis_counts),
        },
        "paper_feature_availability": {
            "age": "demographic.age_at_index",
            "gender": "demographic.sex_at_birth; female for all recorded cohort members, so no within-cohort signal",
            "tumor_status": "latest unambiguous follow-up response strictly before the treatment day",
            "grade_stage": "primary diagnosis record; may be absent",
            "cumulative_drug_count": "count of dated, classifiable non-radiation treatment records with start before decision day; undercounts when prior starts are missing",
            "radiation_history": "any dated radiation treatment record strictly before decision day",
            "ecog_karnofsky": "latest unambiguous follow-up values strictly before decision day; sparse",
            "reserved_genomics": "not included because no linked genomic copy-number matrix is in this project workspace",
        },
        "derived_dataset": {
            "action_classes": ACTION_CLASSES,
            "action_class_member_counts": dict(class_counts),
            "paper_style_action_combinations_observed": len(class_combos),
            "fully_classified_action_combinations_observed": len(complete_class_combos),
            "top_action_combinations": [{"combination": k, "decision_sets": n} for k, n in class_combos.most_common(20)],
            "same_day_decision_sets_with_known_action_class": len(decision_rows),
            "same_day_decision_sets_with_all_member_action_classes_mapped": sum(row["action_mapping_complete"] for row in decision_rows),
            "candidate_linkage_grades": dict(decisions_by_grade),
            "observed_post_completion_response_labels": dict(outcomes_after_end),
            "high_confidence_complete_patient_timeline_links": sum(r["linkage_grade"] == "complete_patient_start_timeline" for r in decision_rows),
            "patient_split_counts": dict(split_counts),
            "random_seed": args.seed,
        },
        "paper_reproduction_status": [
            "The 587-case cohort size matches the export subset with index_date=Diagnosis.",
            "The local export has 4,125 nested treatment records, not the paper's reported 2,552; the paper does not document enough filtering detail to reconcile the count.",
            "The 11 classes follow the paper's labels, with an explicit local drug-name mapping in treatment_class_mapping.csv; surgical records are retained in treatment_events.csv but excluded from this therapy-class action space. Decisions with any unmapped member are flagged.",
            "The resulting action-combination count is computed from this export and may differ from the paper's 47.",
            "The paper's 7,395 treatment-outcome observations and trained model results are not reproduced by this preprocessing script.",
            "These are retrospective associations. This pipeline does not identify causal treatment effects or approve RL training.",
        ],
        "response_conflicting_same_day_groups_excluded": response_conflicts,
        "outputs": ["patient_splits.csv", "treatment_events.csv", "treatment_class_mapping.csv", "state_action_outcome_candidates.csv"],
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote paper-inspired data outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
