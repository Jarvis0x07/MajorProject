#!/usr/bin/env python3
"""Experimental rollout environment using the saved Transformer ensemble.

This is an integration sandbox, not an RL trainer or a clinical treatment
simulator. It samples starting histories and supported action combinations
from the training split, and predicts next recorded-decision states. If the
predicted tumor status is missing, reward is unavailable; otherwise the model's
tumor-free state component is returned only as an explicitly unvalidated proxy.
Do not interpret policy comparisons or rewards as treatment effects.

Example:
  py scripts/transformer_twin_rl_environment.py \
      --input data/processed/paper_method/state_action_outcome_candidates.csv \
      --checkpoint data/processed/transformer_twin/transformer_ensemble.pt \
      --output data/processed/transformer_twin/rl_environment_smoke.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

import transformer_digital_twin as twin


class TransformerTwinEnvironment:
    """Small Gym-like interface around the frozen Transformer ensemble."""

    def __init__(self, rows: list[dict[str, str]], checkpoint_path: Path,
                 min_action_support: int = 5, max_steps: int = 3, seed: int = 42):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if checkpoint.get("state_features") != list(twin.FEATURE_NAMES):
            raise ValueError("Checkpoint state schema does not match Transformer code")
        if checkpoint.get("action_classes") != list(twin.ACTION_CLASSES):
            raise ValueError("Checkpoint action schema does not match Transformer code")
        self.sequence_length = int(checkpoint["sequence_length"])
        self.models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = twin.TransformerDynamics(
                len(twin.FEATURE_NAMES), len(twin.ACTION_CLASSES), self.sequence_length,
                checkpoint["d_model"], checkpoint["heads"], checkpoint["layers"], checkpoint["dropout"],
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
        if len(self.models) < 2:
            raise ValueError("Expected the trained ensemble checkpoint, containing at least two models")

        eligible = [row for row in rows if row.get("split") == "train"
                    and row.get("action_mapping_complete") == "1"
                    and row.get("case_id") and twin.number(row.get("decision_day")) is not None]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        action_counts: Counter[tuple[float, ...]] = Counter()
        for row in eligible:
            grouped[row["case_id"]].append(row)
            action_counts[tuple(twin.encode_action(row))] += 1
        self.patient_rows = {}
        for patient, patient_rows in grouped.items():
            patient_rows.sort(key=lambda row: float(twin.number(row["decision_day"]) or 0.0))
            self.patient_rows[patient] = patient_rows
        self.start_points = [(patient, index) for patient, patient_rows in self.patient_rows.items()
                             for index in range(len(patient_rows))]
        self.supported_actions = [action for action, count in sorted(action_counts.items())
                                  if count >= min_action_support]
        self.action_counts = action_counts
        if not self.start_points or not self.supported_actions:
            raise ValueError("Training split has no eligible start points or supported actions")
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.step_count = 0
        self.states: list[list[float]] = []
        self.prior_actions: list[list[float]] = []
        self.last_transition: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        patient, index = self.rng.choice(self.start_points)
        history = self.patient_rows[patient][:index + 1]
        self.states = [twin.encode_state(row) for row in history]
        self.prior_actions = [twin.encode_action(row) for row in history[:-1]]
        self.step_count = 0
        self.last_transition = None
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        current = self.states[-1]
        return {"features": dict(zip(twin.FEATURE_NAMES, current)),
                "history_length": len(self.states),
                "step": self.step_count}

    def step(self, action_index: int) -> dict[str, Any]:
        if self.step_count >= self.max_steps:
            raise RuntimeError("Episode is complete; call reset() before stepping again")
        if not isinstance(action_index, int) or not 0 <= action_index < len(self.supported_actions):
            raise ValueError("action_index must identify a training-supported action combination")
        action = list(self.supported_actions[action_index])
        rollout = twin.simulate_trajectory(
            self.models, self.states, self.prior_actions, [action],
            self.sequence_length, self.device,
        )
        forecast = rollout["trajectory"][0]
        next_state = forecast["predicted_state"]
        next_vector = [float(next_state[name]) for name in twin.FEATURE_NAMES]
        # Do not turn an unknown tumor state into a positive reward. The model
        # predicts tumor_missing separately; if it remains missing, no response
        # reward can be formed from this state. Even a non-missing score is only
        # a model proxy, not an observed outcome or treatment effect.
        outcome_missing = next_vector[3] >= 0.5
        proxy_reward = None if outcome_missing else float(next_vector[2])
        self.states.append(next_vector)
        self.prior_actions.append(action)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        result = {
            "observation": self._observation(),
            "reward": proxy_reward,
            "terminated": done,
            "truncated": False,
            "action_index": action_index,
            "action_vector": action,
            "action_support_count_in_train": int(self.action_counts[tuple(action)]),
            "ensemble_variance": float(forecast["mean_ensemble_variance"]),
            "reward_semantics": (
                "unavailable_predicted_tumor_status_missing" if outcome_missing else
                "predicted_tumor_free_state_component; unvalidated model proxy, not clinical utility"
            ),
            "warning": rollout["warning"],
        }
        self.last_transition = result
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/paper_method/state_action_outcome_candidates.csv")
    parser.add_argument("--checkpoint", default="data/processed/transformer_twin/transformer_ensemble.pt")
    parser.add_argument("--output", default="data/processed/transformer_twin/rl_environment_smoke.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--min-action-support", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.min_action_support < 1:
        parser.error("episodes and minimum action support must be positive")

    with Path(args.input).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    env = TransformerTwinEnvironment(
        rows, Path(args.checkpoint), args.min_action_support, args.max_steps, args.seed,
    )
    episodes = []
    for episode_index in range(args.episodes):
        initial = env.reset(seed=args.seed + episode_index)
        transitions = []
        for _ in range(args.max_steps):
            action_index = env.rng.randrange(len(env.supported_actions))
            transition = env.step(action_index)
            transitions.append(transition)
            if transition["terminated"]:
                break
        episodes.append({
            "episode": episode_index + 1,
            "initial_observation": initial,
            "transitions": transitions,
            "sum_available_model_proxy_reward": sum(
                item["reward"] for item in transitions if item["reward"] is not None
            ),
            "unavailable_reward_steps": sum(item["reward"] is None for item in transitions),
        })

    result = {
        "environment": "frozen Transformer ensemble rollout sandbox",
        "device": str(env.device),
        "policy": "uniform random over training-supported action combinations; no RL policy trained",
        "supported_action_count": len(env.supported_actions),
        "min_action_support": args.min_action_support,
        "episodes": episodes,
        "limits": [
            "The dynamics model forecasts the next recorded decision state, not an immediate treatment outcome.",
            "Candidate actions are observationally supported in training, not clinically safe or causally evaluated.",
            "Reward is the model-predicted tumor-free state component, not observed utility or a validated outcome model.",
            "This smoke run does not train, evaluate, or recommend an RL policy.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    available_rewards = [item["reward"] for episode in episodes for item in episode["transitions"]
                         if item["reward"] is not None]
    print(json.dumps({"output": str(output_path.resolve()), "device": str(env.device),
                      "supported_action_count": len(env.supported_actions),
                      "episode_count": len(episodes), "transitions": sum(len(e["transitions"]) for e in episodes),
                      "unavailable_reward_steps": sum(e["unavailable_reward_steps"] for e in episodes),
                      "rl_policy_trained": False,
                      "mean_available_model_proxy_reward": (
                          sum(available_rewards) / len(available_rewards) if available_rewards else None
                      )}, indent=2))


if __name__ == "__main__":
    main()
