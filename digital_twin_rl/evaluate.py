"""Evaluate treatment policies on held-out patients' twins.

Each policy treats the twins of test-split patients (never seen by the agent),
starting from their first observed line and, separately, from their first
relapse. Rollouts share seeds across policies (common random numbers) so
policy differences are paired. Policies are scored both inside the twin the
agent trained on and inside an "evaluation twin" built from ensemble members
the agent never interacted with; a large gap between the two signals that the
agent is exploiting model error rather than learning a robust strategy.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .environment import TwinTreatmentEnv

SUPPORT_THRESHOLD = 0.3
METRICS = ["return", "survival_months_5y", "alive_at_5y", "first_line_response",
           "severe_side_effects", "lines_given", "clinician_supported_actions"]


def run_episode(env: TwinTreatmentEnv, policy, start_index: int, seed: int, behavior) -> dict:
    obs, info = env.reset(seed=seed, start_index=start_index)
    total = months = 0.0
    side_effects, actions, supported = 0, [], []
    first_response, died = None, False
    while True:
        action = policy.act(obs, info)
        probs = behavior.probabilities(info["state"])
        supported.append(probs[action] / max(probs.max(), 1e-12) >= SUPPORT_THRESHOLD)
        obs, reward, terminated, truncated, info = env.step(action)
        outcome = info["outcome"]
        total += reward
        months += outcome["months_on_line"]
        side_effects += len(outcome["severe_side_effects"])
        actions.append(outcome["regimen"])
        first_response = outcome["response"] if first_response is None else first_response
        died = died or outcome["died"]
        if terminated or truncated:
            break
    return {"return": total, "survival_months_5y": months, "alive_at_5y": float(not died),
            "first_line_response": float(first_response), "severe_side_effects": float(side_effects),
            "lines_given": float(len(actions)), "clinician_supported_actions": float(np.mean(supported)),
            "actions": actions}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> list[float]:
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate_policies(env_factory, policies: dict, behavior, rollouts: int = 30,
                      seed: int = 7, reference: str = "clinician") -> dict:
    """env_factory(bcq_threshold) -> env; policies: name -> (policy, bcq_threshold)."""
    rng = np.random.default_rng(seed)
    per_policy, per_start = {}, {}
    for name, (policy, bcq) in policies.items():
        env = env_factory(bcq)
        starts = env.start_points()
        rows = []
        first_actions, all_actions = Counter(), Counter()
        for s in range(len(starts)):
            episodes = [run_episode(env, policy, s, seed + 1000 * s + r, behavior) for r in range(rollouts)]
            rows.append({m: float(np.mean([e[m] for e in episodes])) for m in METRICS})
            for e in episodes:
                first_actions[e["actions"][0]] += 1
                all_actions.update(e["actions"])
        per_start[name] = {m: np.array([row[m] for row in rows]) for m in METRICS}
        per_policy[name] = {
            "patients": len(starts),
            "metrics": {m: {"mean": float(per_start[name][m].mean()), "ci95": bootstrap_ci(per_start[name][m], rng)}
                        for m in METRICS},
            "first_action_share": {k: v / sum(first_actions.values()) for k, v in first_actions.most_common()},
            "all_action_share": {k: v / sum(all_actions.values()) for k, v in all_actions.most_common(8)},
        }
    if reference in per_start:
        for name in per_policy:
            if name == reference:
                continue
            diffs = {}
            for m in ("return", "survival_months_5y", "severe_side_effects", "first_line_response"):
                d = per_start[name][m] - per_start[reference][m]
                diffs[m] = {"mean": float(d.mean()), "ci95": bootstrap_ci(d, rng)}
            per_policy[name][f"paired_difference_vs_{reference}"] = diffs
    return per_policy


def plot_results(results: dict, training_history: dict, figure_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in training_history.items():
        ax.plot([h["episode"] for h in history], [h["mean_return"] for h in history], label=name)
    ax.set_xlabel("training episode")
    ax.set_ylabel("mean return (moving window)")
    ax.set_title("RL doctor learning curve inside the training twin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    written.append("learning_curve.png")

    panels = [("survival_months_5y", "survival months (5-year horizon)"),
              ("first_line_response", "response rate (first treated line)"),
              ("severe_side_effects", "severe side effects per patient")]
    for scenario, by_policy in results.items():
        names = list(by_policy)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (metric, label) in zip(axes, panels):
            means = [by_policy[n]["metrics"][metric]["mean"] for n in names]
            lows = [by_policy[n]["metrics"][metric]["ci95"][0] for n in names]
            highs = [by_policy[n]["metrics"][metric]["ci95"][1] for n in names]
            err = [np.subtract(means, lows), np.subtract(highs, means)]
            ax.bar(names, means, yerr=err, capsize=4, color="#4C72B0")
            ax.set_title(label, fontsize=10)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"Held-out patient twins: {scenario.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"policy_comparison_{scenario}.png", dpi=150)
        plt.close(fig)
        written.append(f"policy_comparison_{scenario}.png")

        regimens = sorted({r for n in names for r in by_policy[n]["first_action_share"]})
        matrix = np.array([[by_policy[n]["first_action_share"].get(r, 0.0) for r in regimens] for n in names])
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(regimens) + 2), 0.5 * len(names) + 2))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(regimens)), [r.replace("_", " ") for r in regimens], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(f"First regimen chosen ({scenario.replace('_', ' ')})")
        fig.colorbar(image, ax=ax, label="share of episodes")
        fig.tight_layout()
        fig.savefig(figure_dir / f"first_regimen_{scenario}.png", dpi=150)
        plt.close(fig)
        written.append(f"first_regimen_{scenario}.png")
    return written
