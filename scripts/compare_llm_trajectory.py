#!/usr/bin/env python3
"""Compare a local Ollama LLM with non-LLM baselines on the fixed TCGA test split.

The LLM sees only index-time profile/primary-diagnosis fields and the last
known response. It never receives the target response or later follow-ups.
"""
import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path

from evaluate_trajectory_baseline import metrics, response_timeline


LABELS = ["TF-Tumor Free", "WT-With Tumor"]
OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "string", "enum": LABELS},
        "evidence_fields": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prediction", "evidence_fields", "limitations"],
    "additionalProperties": False,
}


def static_profile(case):
    demo = case.get("demographic") or {}
    primary = next((d for d in case.get("diagnoses") or []
                    if str(d.get("classification_of_tumor", "")).strip().lower() == "primary"), {})
    return {
        "age_at_index_years": demo.get("age_at_index"),
        "sex_at_birth": demo.get("sex_at_birth"),
        "primary_site": case.get("primary_site"),
        "primary_diagnosis": primary.get("primary_diagnosis"),
        "figo_stage": primary.get("figo_stage"),
        "tumor_grade": primary.get("tumor_grade"),
    }


def call_ollama(base_url, model, payload):
    prompt = (
        "Forecast the category recorded at this patient's next follow-up. This is a retrospective research benchmark, "
        "not a treatment decision. Predict exactly one permitted value from disease_response. Do not recommend treatment. "
        "Use only this time-t patient snapshot; no future records or treatment changes are supplied. "
        "Return the required JSON object.\nSnapshot:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": OLLAMA_SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    output = json.loads(result["response"])
    pred = output.get("prediction")
    if pred not in LABELS:
        raise ValueError(f"Model returned invalid response category: {pred!r}")
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="medgemma1.5:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", default="data/processed/llm_trajectory_comparison.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    patients = {}
    case_by_id = {}
    for case in cases:
        if case.get("index_date") != "Diagnosis":
            continue
        patient_id = case.get("case_id") or case.get("submitter_id")
        timeline = response_timeline(case)
        pairs = [{"previous_day": prev[0], "previous_response": prev[1],
                  "target": nxt[1]} for prev, nxt in zip(timeline, timeline[1:]) if nxt[0] > prev[0]]
        if patient_id and pairs:
            patients[patient_id] = pairs
            case_by_id[patient_id] = case
    ids = sorted(patients)
    random.Random(args.seed).shuffle(ids)
    n_test = max(1, round(len(ids) * args.test_fraction))
    test_ids = set(ids[:n_test])
    train_pairs = [p for pid in patients if pid not in test_ids for p in patients[pid]]
    test_pairs = [(pid, p) for pid in patients if pid in test_ids for p in patients[pid]]
    if not train_pairs or not test_pairs:
        raise SystemExit("Not enough eligible patients for train/test comparison")

    majority_counts = {}
    for pair in train_pairs:
        majority_counts[pair["target"]] = majority_counts.get(pair["target"], 0) + 1
    majority_label = sorted(majority_counts, key=lambda x: (-majority_counts[x], x))[0]
    actual = [pair["target"] for _, pair in test_pairs]
    persistence = [pair["previous_response"] for _, pair in test_pairs]
    majority = [majority_label] * len(test_pairs)
    llm_predictions = []
    errors = []
    for index, (pid, pair) in enumerate(test_pairs, start=1):
        case = case_by_id[pid]
        snapshot = {
            "profile_at_diagnosis": static_profile(case),
            "last_known_follow_up": {
                "day_since_diagnosis": pair["previous_day"],
                "disease_response": pair["previous_response"],
            },
            "forecast_target": "disease_response at next recorded follow-up",
            "allowed_categories": LABELS,
        }
        try:
            llm_predictions.append(call_ollama(args.ollama_url, args.model, snapshot))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            errors.append({"test_transition": index, "error": str(exc)})
            llm_predictions.append("__ABSTAIN__")

    compared_labels = LABELS
    result = {
        "study": "Zero-shot local LLM forecast of next recorded follow-up response",
        "model": args.model,
        "split": {"seed": args.seed, "test_fraction": args.test_fraction,
                  "test_patients": len(test_ids), "test_transitions": len(test_pairs)},
        "test_target_support": {label: actual.count(label) for label in compared_labels},
        "llm_valid_predictions": sum(p in LABELS for p in llm_predictions),
        "llm_abstentions_or_errors": len(errors),
        "errors": errors,
        "metrics": {
            "last_response_persistence": metrics(actual, persistence, compared_labels),
            "training_majority_class": metrics(actual, majority, compared_labels),
            "llm_zero_shot": metrics(actual, llm_predictions, compared_labels),
        },
        "limitations": [
            "Small held-out set; this single split is exploratory.",
            "LLM input excludes treatment history to avoid unresolved event-linkage and treatment-effect assumptions.",
            "Forecast is an observational response-category prediction, not a counterfactual treatment effect.",
            "Self-reported model rationales/confidence are not treated as validated evidence or calibrated probabilities.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved comparison to {out.resolve()}")
    if errors:
        raise SystemExit("Some LLM calls failed or returned invalid output; inspect the saved report.")


if __name__ == "__main__":
    main()
