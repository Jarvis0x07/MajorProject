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

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .environment import TwinTreatmentEnv, safety_mask
from .twin import PatientDigitalTwin

SUPPORT_THRESHOLD = 0.3
FIDELITY_MONTHS = (6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0)
METRICS = ["return", "survival_months_5y", "alive_at_5y", "first_line_response",
           "severe_side_effects", "lines_given", "clinician_supported_actions",
           "line_limit_reached", "mask_fallback_steps"]


def run_episode(env: TwinTreatmentEnv, policy, start_index: int, seed: int, behavior) -> dict:
    """One treated twin. survival_months_5y = simulated months alive within the 5-year window."""
    obs, info = env.reset(seed=seed, start_index=start_index)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    total = months = 0.0
    side_effects, actions, supported, fallbacks = 0, [], [], 0
    first_response, died, truncated = None, False, False
    while True:
        action = policy.act(obs, info)
        probs = behavior.probabilities(info["state"])
        supported.append(probs[action] / max(probs.max(), 1e-12) >= SUPPORT_THRESHOLD)
        fallbacks += int(any(info["mask_flags"].values()))
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
            "line_limit_reached": float(truncated), "mask_fallback_steps": float(fallbacks),
            "actions": actions}


def kaplan_meier(times: np.ndarray, events: np.ndarray, at: tuple[float, ...]) -> dict[str, float]:
    """Kaplan-Meier survival probabilities at the requested months."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    survival, curve = 1.0, []
    at_risk = len(times)
    for t in np.unique(times):
        here = times == t
        deaths = int(events[here].sum())
        if deaths:
            survival *= 1.0 - deaths / at_risk
        curve.append((t, survival))
        at_risk -= int(here.sum())
    result = {}
    for m in at:
        before = [s for t, s in curve if t <= m]
        result[f"{int(m)}m"] = float(before[-1]) if before else 1.0
    return result


def simulation_fidelity(histories: list[dict], ensemble, toxicity, behavior, regimens: list[list[str]],
                        members: list[int], split: str = "test", rollouts: int = 30, seed: int = 7) -> dict:
    """Replay each held-out patient's observed regimens in their twin; compare survival with reality.

    Starting at the first observed line, the twin receives the regimens the
    patient actually received; once those run out, the clinician model chooses.
    Simulated survival from first line is compared with the Kaplan-Meier
    estimate of the patients' observed overall survival.
    """
    patients = [p for p in histories if p["split"] == split and p["lines"]]
    observed = kaplan_meier(np.array([p["lines"][0]["outcome"]["os_months"] for p in patients]),
                            np.array([p["lines"][0]["outcome"]["os_event"] for p in patients]), FIDELITY_MONTHS)
    alive = {f"{int(m)}m": [] for m in FIDELITY_MONTHS}
    for k, patient in enumerate(patients):
        for r in range(rollouts):
            rng = random.Random(seed + 1000 * k + r)
            twin = PatientDigitalTwin(patient, ensemble, toxicity, 0, members)
            line = 0
            while twin.end_reason is None and line < 12:
                if line < len(patient["lines"]):
                    regimen = patient["lines"][line]["regimen"]
                else:
                    probs = np.where(safety_mask(twin.state, regimens), behavior.probabilities(twin.state), -1.0)
                    regimen = regimens[int(np.argmax(probs))]
                twin.step(regimen, rng)
                line += 1
            death_time = twin.months_elapsed if twin.end_reason == "death" else math.inf
            for m in FIDELITY_MONTHS:
                alive[f"{int(m)}m"].append(death_time > m)
    simulated = {k: float(np.mean(v)) for k, v in alive.items()}
    return {"patients": len(patients), "rollouts_per_patient": rollouts,
            "observed_kaplan_meier_survival": observed, "simulated_survival": simulated,
            "note": "Observed survival is from the first recorded treatment line; the twin replays the observed regimens."}


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


def plot_results(results: dict, training_history: dict, figure_dir: Path, fidelity: dict | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if fidelity:
        months = [0] + [int(k[:-1]) for k in fidelity["observed_kaplan_meier_survival"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(months, [1.0] + list(fidelity["observed_kaplan_meier_survival"].values()), "s-",
                label="observed (Kaplan-Meier)", color="#222222")
        ax.plot(months, [1.0] + list(fidelity["simulated_survival"].values()), "o--",
                label="patient twins replaying observed regimens", color="#4C72B0")
        ax.set_xlabel("months from first treatment line")
        ax.set_ylabel("proportion alive")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Simulation fidelity, {fidelity['patients']} held-out patients")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "simulation_fidelity.png", dpi=150)
        plt.close(fig)
        written.append("simulation_fidelity.png")

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
