#!/usr/bin/env python3
"""Experimental LLM digital-twin environment and Objective 3 policy baselines.

This is a research harness, not a trained clinical simulator or treatment policy.
It reuses paper-inspired patient states/actions, asks a configured local Ollama
LLM for a structured next-response hypothesis, and compares simple policies in
that same hypothesis generator. TCGA-OV lacks sufficient complete action/outcome
links, toxicity, cost, and quality-of-life labels; simulated rollouts therefore
must not be interpreted as treatment effects or clinical recommendations.

Example (requires a running local Ollama server and a pulled model):
  python scripts/llm_twin_rl.py --input data/processed/paper_method/state_action_outcome_candidates.csv \
      --model medgemma1.5:4b --episodes 20 --experimental-counterfactual
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESPONSE_LABELS = ("TF-Tumor Free", "WT-With Tumor")
STATE_FIELDS = (
    "state_age_years", "state_gender", "state_tumor_status_pre_action",
    "state_tumor_grade", "state_figo_stage",
    "state_cumulative_dated_drug_record_count", "state_radiation_history",
    "state_ecog_pre_action", "state_karnofsky_pre_action",
    "state_has_undated_treatment_records",
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "next_response": {"type": ["string", "null"], "enum": [*RESPONSE_LABELS, None]},
        "evidence_fields": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["next_response", "evidence_fields", "limitations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RewardComponents:
    """Only observed/predicted response is available in this cohort.

    The non-response terms are intentionally nullable. A nonzero configured
    weight for a missing term makes reward unavailable instead of imputing it.
    """

    response: float | None
    toxicity_penalty: float | None = None
    cost_penalty: float | None = None
    wellbeing: float | None = None


@dataclass(frozen=True)
class RewardWeights:
    effectiveness: float = 1.0
    toxicity: float = 0.0
    cost: float = 0.0
    wellbeing: float = 0.0


def calculate_reward(components: RewardComponents, weights: RewardWeights) -> float | None:
    pairs = (
        (components.response, weights.effectiveness),
        (components.toxicity_penalty, weights.toxicity),
        (components.cost_penalty, weights.cost),
        (components.wellbeing, weights.wellbeing),
    )
    if any(value is None and weight != 0 for value, weight in pairs):
        return None
    return sum(float(value) * weight for value, weight in pairs if value is not None)


class OllamaResponseTwin:
    """Strict JSON adapter. It predicts an observational response category only."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, state: dict[str, Any], action: str, step_number: int, history: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "current_state": {key: state.get(key) for key in STATE_FIELDS},
            "candidate_treatment_action": action,
            "decision_step": step_number,
            "previous_simulated_steps": history,
            "available_response_categories": list(RESPONSE_LABELS),
            "unavailable_outcomes": ["toxicity", "treatment complications", "cost", "quality of life", "survival"],
        }
        prompt = (
            "You are a research-only outcome hypothesis generator in an experimental ovarian-cancer digital-twin study. "
            "Given the structured patient snapshot and candidate historical treatment class, return a JSON hypothesis for "
            "the next recorded tumor-response category. This is NOT a causal estimate, medical advice, or a real patient "
            "recommendation. Do not infer toxicity, cost, quality of life, survival, dose, or unprovided clinical facts. "
            "If the supplied data cannot support even a weak category hypothesis, set next_response to null. "
            "Use only the two allowed categories or null; list only keys present in the snapshot as evidence_fields; include "
            "the limitations that timing is irregular, histories may be incomplete, and the output is not clinically validated.\n"
            "Return exactly the required JSON object. Snapshot:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        request_body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "format": OUTPUT_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/generate", data=request_body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        result = json.loads(raw["response"])
        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object")
        label = result.get("next_response")
        if label is not None and label not in RESPONSE_LABELS:
            raise ValueError(f"Unsupported response category: {label!r}")
        for name in ("evidence_fields", "limitations"):
            if not isinstance(result.get(name), list) or not all(isinstance(x, str) for x in result[name]):
                raise ValueError(f"{name} must be an array of strings")
        allowed_evidence = set(STATE_FIELDS) | {"candidate_treatment_action", "previous_simulated_steps"}
        if any(field not in allowed_evidence for field in result["evidence_fields"]):
            raise ValueError("LLM cited a field not provided to it")
        return result


class TreatmentSimulationEnvironment:
    """Small event-based, LLM-generated simulation; each rollout is hypothetical."""

    def __init__(self, start_rows: list[dict[str, str]], supported_actions: list[str], twin: OllamaResponseTwin,
                 max_steps: int = 3, gamma: float = 0.95):
        if not start_rows or not supported_actions:
            raise ValueError("Need training-split patient contexts and supported historical actions")
        self.start_rows = start_rows
        self.supported_actions = supported_actions
        self.twin = twin
        self.max_steps = max_steps
        self.gamma = gamma
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.step_number = 0
        self.rng = random.Random(42)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        row = self.rng.choice(self.start_rows)
        self.state = {field: row.get(field) or None for field in STATE_FIELDS}
        self.history = []
        self.step_number = 0
        return dict(self.state)

    def step(self, action: str) -> dict[str, Any]:
        if action not in self.supported_actions:
            raise ValueError("Action is not in the empirically supported action set")
        if self.step_number >= self.max_steps:
            raise RuntimeError("Episode is already complete")
        prediction = self.twin.predict(self.state, action, self.step_number + 1, self.history)
        response = prediction["next_response"]
        response_reward = None if response is None else 1.0 if response == "TF-Tumor Free" else 0.0
        reward_components = RewardComponents(response=response_reward)
        reward = calculate_reward(reward_components, RewardWeights())
        transition = {
            "action": action,
            "predicted_response": response,
            "response_proxy_reward": reward,
            "reward_components": asdict(reward_components),
            "toxicity_cost_wellbeing": "unavailable_in_dataset",
            "evidence_fields": prediction["evidence_fields"],
            "limitations": prediction["limitations"],
        }
        self.history.append({"action": action, "predicted_response": response or "abstain"})
        if response is not None:
            self.state["state_tumor_status_pre_action"] = response
        self.step_number += 1
        done = response is None or self.step_number >= self.max_steps
        return {"state": dict(self.state), "reward": reward, "done": done, "transition": transition}


def load_training_data(path: Path, min_action_support: int) -> tuple[list[dict[str, str]], list[str], Counter[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Only train partition states seed rollouts. Do not use observed outcomes as
    # prompt features. Candidate quality is preserved in reports but cannot make
    # a counterfactual prediction factual.
    train_rows = [row for row in rows if row.get("split") == "train" and row.get("action_mapping_complete") == "1"]
    supports = Counter(row.get("action_combination", "") for row in train_rows if row.get("action_combination"))
    actions = sorted(action for action, count in supports.items() if count >= min_action_support)
    if not train_rows:
        raise ValueError("No fully mapped training-split states found")
    if not actions:
        raise ValueError("No actions meet --min-action-support; lower threshold only for exploratory software runs")
    return train_rows, actions, supports


def validate_observed_holdout(path: Path, twin: OllamaResponseTwin, supported_actions: list[str], limit: int) -> dict[str, Any]:
    """Predict factual observed-action outcomes on test patients, not counterfactuals."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def eligible_row(row: dict[str, str], split: str) -> bool:
        return (row.get("split") == split
                and row.get("action_mapping_complete") == "1"
                and row.get("linkage_grade") == "dated_event_only_candidate"
                and row.get("next_response_after_completion") in RESPONSE_LABELS
                and row.get("action_combination") in supported_actions)

    eligible = [row for row in rows if eligible_row(row, "test")]
    eligible.sort(key=lambda row: (row.get("case_id", ""), row.get("decision_day", "")))
    if limit > 0:
        eligible = eligible[:limit]
    training_targets = [row["next_response_after_completion"] for row in rows
                        if eligible_row(row, "train")]
    if not training_targets:
        majority_label = None
    else:
        counts = Counter(training_targets)
        majority_label = min(counts, key=lambda label: (-counts[label], label))

    records: list[dict[str, Any]] = []
    request_errors: list[dict[str, Any]] = []
    for row_number, row in enumerate(eligible, start=1):
        state = {field: row.get(field) or None for field in STATE_FIELDS}
        predicted = None
        error = None
        try:
            prediction = twin.predict(state, row["action_combination"], 1, [])
            predicted = prediction["next_response"]
        except Exception as exc:  # Keep individual endpoint/format failures visible in the report.
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            request_errors.append({"row_number": row_number, **error})
        previous = row.get("state_tumor_status_pre_action")
        records.append({
            "row_number": row_number,
            "decision_day": row.get("decision_day"),
            "action": row.get("action_combination"),
            "actual_response": row["next_response_after_completion"],
            "llm_prediction": predicted,
            "persistence_prediction": previous if previous in RESPONSE_LABELS else None,
            "training_majority_prediction": majority_label,
            "request_error": error,
        })

    def metrics_for(items: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
        valid = [item for item in items if item.get(prediction_key) in RESPONSE_LABELS]
        recalls = {}
        for label in RESPONSE_LABELS:
            class_items = [item for item in valid if item["actual_response"] == label]
            recalls[label] = (sum(item[prediction_key] == label for item in class_items) / len(class_items)
                              if class_items else None)
        supported_recalls = [value for value in recalls.values() if value is not None]
        correct = sum(item[prediction_key] == item["actual_response"] for item in valid)
        return {
            "scored_rows": len(valid),
            "coverage": len(valid) / len(items) if items else None,
            "accuracy_on_scored_rows": correct / len(valid) if valid else None,
            "accuracy_with_abstentions_wrong": correct / len(items) if items else None,
            "balanced_accuracy_on_scored_rows": sum(supported_recalls) / len(supported_recalls) if supported_recalls else None,
            "class_support_on_scored_rows": {label: sum(item["actual_response"] == label for item in valid)
                                              for label in RESPONSE_LABELS},
            "recall_by_class": recalls,
        }

    # Compare all predictors on identical rows where both the LLM and the
    # persistence baseline have valid outputs. The majority baseline is learned
    # only from eligible training-split response labels.
    paired = [item for item in records
              if item["llm_prediction"] in RESPONSE_LABELS
              and item["persistence_prediction"] in RESPONSE_LABELS]
    return {
        "evaluation": "factual observed-action predictions on held-out test patients",
        "evaluated_rows": len(eligible),
        "unique_patients": len({row.get("case_id") for row in eligible if row.get("case_id")}),
        "training_majority_label": majority_label,
        "actual_label_counts": {label: sum(item["actual_response"] == label for item in records)
                                for label in RESPONSE_LABELS},
        "per_model_metrics": {
            "llm": metrics_for(records, "llm_prediction"),
            "last_response_persistence": metrics_for(records, "persistence_prediction"),
            "training_majority": metrics_for(records, "training_majority_prediction"),
        },
        "paired_comparison": {
            "description": "All three predictions scored on the same rows where LLM and persistence both returned permitted labels.",
            "rows": len(paired),
            "llm": metrics_for(paired, "llm_prediction"),
            "last_response_persistence": metrics_for(paired, "persistence_prediction"),
            "training_majority": metrics_for(paired, "training_majority_prediction"),
        },
        "request_errors": request_errors,
        "row_predictions": records,
        "limitations": [
            "This evaluates observed action/outcome pairs, not alternative-treatment effects.",
            "Undated treatments can still hide intervening actions in dated_event_only_candidate records.",
            "The target is the next qualifying recorded response, not a fixed-horizon response.",
            "Small support and class imbalance can make the estimate unstable; compare balanced accuracy and class recall, not accuracy alone.",
        ],
    }


class BaselinePolicies:
    """Action-support baselines; none is a clinical standard-of-care policy."""

    def __init__(self, supports: Counter[str], actions: list[str], seed: int = 42):
        self.supports = supports
        self.actions = actions
        self.rng = random.Random(seed)
        self.overall = max(actions, key=lambda a: (supports[a], a))
        by_status: dict[str, Counter[str]] = defaultdict(Counter)
        # State-conditional frequencies are from training rows only.
        self.by_status = by_status

    def bind_rows(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            action, status = row.get("action_combination"), row.get("state_tumor_status_pre_action") or "<missing>"
            if action in self.actions:
                self.by_status[status][action] += 1

    def choose(self, name: str, state: dict[str, Any]) -> str:
        if name == "random_supported":
            return self.rng.choice(self.actions)
        if name == "historical_most_common":
            return self.overall
        if name == "status_conditioned_historical":
            status = state.get("state_tumor_status_pre_action") or "<missing>"
            counts = self.by_status.get(status)
            if counts:
                return max(counts, key=lambda a: (counts[a], a))
            return self.overall
        raise ValueError(f"Unknown baseline: {name}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows, actions, supports = load_training_data(Path(args.input), args.min_action_support)
    policies = BaselinePolicies(supports, actions, args.seed)
    policies.bind_rows(rows)
    twin = OllamaResponseTwin(args.model, args.ollama_url, args.timeout)
    episodes: dict[str, list[dict[str, Any]]] = {name: [] for name in (
        "random_supported", "historical_most_common", "status_conditioned_historical")}
    # Same initial training contexts make policy comparisons less noisy.
    random.seed(args.seed)
    initial_seeds = [random.randrange(0, 2**31) for _ in range(args.episodes)]
    for policy_name in episodes:
        env = TreatmentSimulationEnvironment(rows, actions, twin, args.max_steps, args.gamma)
        for seed in initial_seeds:
            state = env.reset(seed)
            trajectory = []
            for _ in range(args.max_steps):
                action = policies.choose(policy_name, state)
                result = env.step(action)
                state = result["state"]
                trajectory.append(result["transition"])
                if result["done"]:
                    break
            episodes[policy_name].append(trajectory)

    metrics: dict[str, Any] = {}
    for name, trajectories in episodes.items():
        labels = [step["predicted_response"] for ep in trajectories for step in ep if step["predicted_response"]]
        rewards = [step["response_proxy_reward"] for ep in trajectories for step in ep if step["response_proxy_reward"] is not None]
        metrics[name] = {
            "episodes": len(trajectories),
            "simulated_steps": sum(map(len, trajectories)),
            "valid_response_hypotheses": len(labels),
            "abstentions": sum(step["predicted_response"] is None for ep in trajectories for step in ep),
            "predicted_tumor_free_fraction": labels.count("TF-Tumor Free") / len(labels) if labels else None,
            "mean_response_proxy_reward": sum(rewards) / len(rewards) if rewards else None,
            "toxicity_cost_wellbeing_evaluation": "not possible: these outcomes are absent from the dataset",
        }
    return {
        "study": "Experimental LLM digital-twin policy-baseline comparison",
        "model": args.model,
        "dataset": str(Path(args.input)),
        "method": "Candidate counterfactual rollouts from training-split states; LLM predicts next recorded response category",
        "mdp": {
            "state_fields": list(STATE_FIELDS),
            "actions": actions,
            "action_support_minimum_training_rows": args.min_action_support,
            "decision_epoch": "one simulated treatment decision; irregular clinical time is not represented",
            "reward": "1 for predicted TF-Tumor Free, 0 for predicted WT-With Tumor; abstention has no reward",
            "gamma": args.gamma,
            "max_steps": args.max_steps,
            "episode_termination": "model abstention or max_steps",
            "treatment_constraints": "class combinations observed at least min_action_support times in training split; no dose actions",
            "safety_constraints": "only dataset-supported action classes; no toxicity, contraindication, or dose-safety model exists",
        },
        "policy_training": "none; these are random and historical-frequency baselines for Objective 3, not an RL-optimized policy",
        "split": {"source": "patient_splits.csv embedded in candidate rows", "seed": args.seed,
                  "rollout_initial_states": args.episodes, "state_seed_partition": "train only"},
        "policy_metrics": metrics,
        "observed_holdout_check": validate_observed_holdout(
            Path(args.input), twin, actions, args.validation_limit
        ) if args.validate_heldout else "not requested",
        "warnings": [
            "Every rollout is a model-generated hypothesis, not an observed or causal treatment effect.",
            "Action support is observational frequency and does not prove clinical appropriateness or treatment overlap.",
            "The candidate initial states include incomplete treatment histories flagged in source data.",
            "No toxicities, costs, quality-of-life values, or calibrated confidence are available; safety and multi-objective utility cannot be evaluated.",
            "Do not use this prototype for patient care or real treatment recommendations.",
        ],
        "episodes": episodes if args.include_trajectories else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/paper_method/state_action_outcome_candidates.csv")
    parser.add_argument("--model", default="medgemma1.5:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", default="data/processed/objectives_2_3/policy_baseline_simulation.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--min-action-support", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experimental-counterfactual", action="store_true",
                        help="Required acknowledgement: outputs are unvalidated counterfactual hypotheses")
    parser.add_argument("--include-trajectories", action="store_true")
    parser.add_argument("--validate-heldout", action="store_true",
                        help="predict factual outcomes for eligible supported-action rows in test patients")
    parser.add_argument("--validation-limit", type=int, default=50,
                        help="maximum held-out rows; 0 means all eligible rows")
    args = parser.parse_args()
    if not args.experimental_counterfactual:
        parser.error("add --experimental-counterfactual to acknowledge the unvalidated research-only rollouts")
    if args.episodes < 1 or args.max_steps < 1 or not 0 <= args.gamma <= 1:
        parser.error("episodes and max-steps must be positive; gamma must be in [0,1]")
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2, ensure_ascii=False))
    print(f"\nSaved simulation report to {output.resolve()}")


if __name__ == "__main__":
    main()
