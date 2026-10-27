#!/usr/bin/env python3
"""Evaluate non-LLM baselines for next observed TCGA-OV response.

Uses only ordered follow-up records with a numeric days_to_follow_up and a
known disease_response. Splits patients before scoring. The persistence model
predicts the next response equals the patient's last observed response; the
majority model predicts the most common training target. This is a forecasting
baseline, not a treatment-effect model or clinical simulator.
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXCLUDED = {"unknown", "not reported", "not available", ""}


def response_timeline(case):
    """Return unambiguous response values at strictly increasing follow-up days."""
    by_day = defaultdict(set)
    for record in case.get("follow_ups") or []:
        day = record.get("days_to_follow_up")
        response = record.get("disease_response")
        if day is None or response is None or str(response).strip().lower() in EXCLUDED:
            continue
        try:
            day = int(day)
        except (ValueError, TypeError):
            continue
        by_day[day].add(str(response).strip())
    # If a timepoint has conflicting responses, omit it rather than inventing
    # an ordering or choosing one label arbitrarily.
    return [(day, next(iter(labels))) for day, labels in sorted(by_day.items()) if len(labels) == 1]


def metrics(actual, predicted, labels):
    n = len(actual)
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / n if n else None
    class_recalls = []
    class_f1s = []
    confusion = {}
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        support = tp + fn
        class_recalls.append(tp / support if support else None)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        class_f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        confusion[label] = {pred: sum(a == label and p == pred for a, p in zip(actual, predicted)) for pred in labels}
    present_recalls = [x for x in class_recalls if x is not None]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (sum(present_recalls) / len(present_recalls)
                              if len(present_recalls) == len(labels) and labels else None),
        "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else None,
        "confusion_matrix_actual_by_predicted": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TCGA-OV JSON export")
    parser.add_argument("--output", default="data/processed/trajectory_baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    patients = {}
    unverified_index_cases = 0
    for case in cases:
        if case.get("index_date") != "Diagnosis":
            unverified_index_cases += 1
            continue
        patient_id = case.get("case_id") or case.get("submitter_id")
        timeline = response_timeline(case)
        pairs = [{"previous_response": a[1], "target": b[1]}
                 for a, b in zip(timeline, timeline[1:]) if b[0] > a[0]]
        if patient_id and pairs:
            patients[patient_id] = pairs

    patient_ids = sorted(patients)
    random.Random(args.seed).shuffle(patient_ids)
    test_count = max(1, round(len(patient_ids) * args.test_fraction))
    test_ids = set(patient_ids[:test_count])
    train_pairs = [pair for pid in patients if pid not in test_ids for pair in patients[pid]]
    test_pairs = [pair for pid in patients if pid in test_ids for pair in patients[pid]]
    if not train_pairs or not test_pairs:
        raise SystemExit("Not enough patients with repeated timed responses for a patient-level split")

    target_counts = Counter(p["target"] for p in train_pairs)
    majority = sorted(target_counts, key=lambda label: (-target_counts[label], label))[0]
    labels = sorted({p["target"] for p in train_pairs + test_pairs})
    actual = [p["target"] for p in test_pairs]
    predictions = {
        "persistence_last_response": [p["previous_response"] for p in test_pairs],
        "training_majority_class": [majority] * len(test_pairs),
    }
    repeated_scores = defaultdict(list)
    repeated_test_transitions = []
    for repeat in range(args.repeats):
        strata = defaultdict(list)
        for pid, pairs in patients.items():
            pattern = tuple(sorted({pair["target"] for pair in pairs}))
            strata[pattern].append(pid)
        held_out = set()
        rng = random.Random(args.seed + repeat)
        for members in strata.values():
            rng.shuffle(members)
            n_stratum_test = min(len(members) - 1, round(len(members) * args.test_fraction))
            if n_stratum_test > 0:
                held_out.update(members[:n_stratum_test])
        training = [pair for pid in patients if pid not in held_out for pair in patients[pid]]
        validation = [pair for pid in patients if pid in held_out for pair in patients[pid]]
        if not training or not validation:
            continue
        majority_counts = Counter(pair["target"] for pair in training)
        majority_label = sorted(majority_counts, key=lambda label: (-majority_counts[label], label))[0]
        truth = [pair["target"] for pair in validation]
        pred_by_model = {
            "persistence_last_response": [pair["previous_response"] for pair in validation],
            "training_majority_class": [majority_label] * len(validation),
        }
        repeated_test_transitions.append(len(validation))
        for model_name, pred in pred_by_model.items():
            score = metrics(truth, pred, labels)
            for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
                if score[metric_name] is not None:
                    repeated_scores[model_name + "." + metric_name].append(score[metric_name])

    repeated_summary = {}
    for key, values in repeated_scores.items():
        repeated_summary[key] = {
            "mean": statistics.mean(values),
            "standard_deviation_across_splits": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
            "splits": len(values),
        }
    results = {
        "study": "Next observed follow-up disease_response category",
        "target": "follow_ups[].disease_response",
        "time_field": "follow_ups[].days_to_follow_up",
        "dataset_cases": len(cases),
        "cases_excluded_without_index_date_diagnosis": unverified_index_cases,
        "patients_with_at_least_one_transition": len(patients),
        "train_patients": len(patient_ids) - len(test_ids),
        "test_patients": len(test_ids),
        "train_transitions": len(train_pairs),
        "test_transitions": len(test_pairs),
        "repeated_patient_stratified_holdout": {
            "requested_repeats": args.repeats,
            "completed_repeats": len(repeated_test_transitions),
            "test_fraction": args.test_fraction,
            "seed_start": args.seed,
            "mean_test_transitions_per_split": statistics.mean(repeated_test_transitions) if repeated_test_transitions else None,
            "metric_distributions": repeated_summary,
        },
        "test_target_support": dict(Counter(actual)),
        "training_majority_label": majority,
        "training_target_support": dict(target_counts),
        "metrics": {name: metrics(actual, pred, labels) for name, pred in predictions.items()},
        "method_notes": [
            "Patient-level split with deterministic seed; no patient occurs in both partitions.",
            "Same-day records with conflicting response labels are excluded; Unknown/missing responses and records without numeric follow-up day are excluded.",
            "Uses only cases whose index_date is Diagnosis; the TCGA follow-up form defines permitted intervals from initial pathological diagnosis.",
            "Repeated results use 100 seeded patient-level stratified 80/20 holdouts; transitions from one patient never cross a split.",
            "The predictor uses the last observed response only; it does not use treatment actions, estimate treatment effects, or support counterfactual claims.",
            "This is a non-LLM forecasting baseline. Performance does not validate the digital twin or imply clinical utility.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
