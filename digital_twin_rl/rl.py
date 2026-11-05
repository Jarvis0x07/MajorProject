"""Step 4: the twins as an environment, and a DQN agent that learns to treat them.

One episode is one patient's twin. At each decision the agent picks a regimen; the twin
simulates that treatment line (response, then progression or death) and moves on to the next
decision, until death or 5 years.

    reward per line = years alive during the line (until progression or death)
                    + 0.25 x chance the tumour responds
                    - 0.5 x disagreement between the twin's networks (do not exploit model error)

Constraint (from treatment guidelines): no platinum when the cancer came back within 6 months
of the last platinum (platinum-resistant disease).

The agent is a deep Q-network. Episodes are only a few lines long, so each Q-value is trained
toward the discounted return actually observed for the rest of the episode (Monte Carlo targets).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import FEATURES, HORIZON, OutcomeModel, encode_state
from .twin import DigitalTwin

MAX_LINES = 8
GAMMA = 0.99
RESPONSE_BONUS = 0.25
UNCERTAINTY_PENALTY = 0.5


def supported_regimens(histories: list[dict], min_lines: int = 10) -> list[list[str]]:
    """The action space: regimens given in at least `min_lines` training lines."""
    counts = Counter(tuple(l["regimen"]) for p in histories if p["split"] == "train" for l in p["lines"])
    return [list(r) for r, n in counts.most_common() if n >= min_lines and "other" not in r]


def platinum_resistant(state: dict) -> bool:
    platinum = state.get("months_since_last_platinum")
    return platinum is not None and platinum < 6.0 and state.get("setting") != "primary"


def allowed(state: dict, regimens: list[list[str]]) -> np.ndarray:
    return np.array([not (platinum_resistant(state) and "platinum" in r) for r in regimens])


def observe(state: dict, months_elapsed: float = 0.0) -> np.ndarray:
    return np.array(encode_state(state) + [months_elapsed / HORIZON], dtype=np.float32)


class TwinEnv:
    def __init__(self, twins: list[dict], model: OutcomeModel, regimens: list[list[str]], seed: int = 0):
        self.twins, self.model, self.regimens = twins, model, regimens
        self.rng = random.Random(seed)
        self.twin: DigitalTwin | None = None

    def reset(self, index: int | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        record = self.twins[index] if index is not None else self.rng.choice(self.twins)
        self.twin = DigitalTwin(record["state"], self.model, record["multipliers"])
        self.lines = 0
        return observe(self.twin.state, 0.0)

    def mask(self) -> np.ndarray:
        return allowed(self.twin.state, self.regimens)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        out = self.twin.step(self.regimens[action], self.rng)
        self.lines += 1
        reward = (out["expected_months"] / 12.0 + RESPONSE_BONUS * out["expected_response"]
                  - UNCERTAINTY_PENALTY * out["uncertainty"])
        done = out["ended"] is not None or self.lines >= MAX_LINES
        return observe(self.twin.state, self.twin.months_elapsed), reward, done, out


def q_network(n_actions: int) -> nn.Module:
    return nn.Sequential(nn.Linear(len(FEATURES) + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, n_actions))


def best_action(network: nn.Module, observation: np.ndarray, mask: np.ndarray) -> int:
    with torch.no_grad():
        q = network(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
    return int(np.argmax(np.where(mask, q, -np.inf)))


def train_agent(env: TwinEnv, episodes: int, seed: int = 42) -> tuple[nn.Module, list[dict]]:
    random.seed(seed)
    torch.manual_seed(seed)
    env.rng.seed(seed)
    batch_rng = np.random.default_rng(seed)
    network = q_network(len(env.regimens))
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-4)
    seen_obs, seen_actions, seen_returns = [], [], []
    history, returns = [], []
    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - 0.95 * episode / (0.6 * episodes))     # explore a lot first, then less
        obs, done, steps = env.reset(), False, []
        while not done:
            mask = env.mask()
            if random.random() < epsilon:
                action = random.choice(np.flatnonzero(mask).tolist())
            else:
                action = best_action(network, obs, mask)
            next_obs, reward, done, _ = env.step(action)
            steps.append((obs, action, reward))
            obs = next_obs
        future = 0.0
        for obs_t, action_t, reward_t in reversed(steps):
            future = reward_t + GAMMA * future
            seen_obs.append(obs_t)
            seen_actions.append(action_t)
            seen_returns.append(future)
        returns.append(sum(r for _, _, r in steps))
        for _ in steps:
            if len(seen_returns) < 1000:
                break
            idx = batch_rng.integers(0, len(seen_returns), 128)
            q = network(torch.from_numpy(np.stack([seen_obs[i] for i in idx])))
            q_taken = q.gather(1, torch.tensor([seen_actions[i] for i in idx]).unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_taken, torch.tensor([seen_returns[i] for i in idx], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (episode + 1) % 1000 == 0:
            history.append({"episode": episode + 1, "mean_return": float(np.mean(returns[-1000:]))})
            print(f"  episode {episode + 1:>6}  exploring {epsilon:.2f}  mean return {history[-1]['mean_return']:.3f}")
    return network.eval(), history


def save_agent(path: Path, network: nn.Module, regimens: list[list[str]]) -> None:
    torch.save({"features": FEATURES, "regimens": regimens, "weights": network.state_dict()}, path)


def load_agent(path: Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["features"] != FEATURES:
        raise ValueError("The saved agent uses different features; run `python -m digital_twin_rl train-agent`")
    network = q_network(len(checkpoint["regimens"]))
    network.load_state_dict(checkpoint["weights"])
    return network.eval()


def standard_care(regimens: list[list[str]]):
    """Baseline doctor: carboplatin + paclitaxel, or liposomal doxorubicin when platinum-resistant."""
    keys = ["+".join(r) for r in regimens]

    def policy(obs: np.ndarray, state: dict, mask: np.ndarray) -> int:
        for key in ("platinum+taxanes", "anthracyclines"):
            if key in keys and mask[keys.index(key)]:
                return keys.index(key)
        return int(np.flatnonzero(mask)[0])
    return policy


def compare_policies(env: TwinEnv, network: nn.Module, rollouts: int = 20, seed: int = 7) -> dict:
    """Months alive within 5 years when each doctor treats the twins of the test patients.

    Every doctor sees the same random draws for the same patient and run, so differences are fair.
    """
    chooser = random.Random(seed)
    policies = {"random": lambda obs, state, mask: chooser.choice(np.flatnonzero(mask).tolist()),
                "standard_care": standard_care(env.regimens),
                "dqn_agent": lambda obs, state, mask: best_action(network, obs, mask)}
    results = {}
    for name, policy in policies.items():
        alive, first = [], Counter()
        for i in range(len(env.twins)):
            for run in range(rollouts):
                obs, done, lines = env.reset(index=i, seed=seed + 1000 * i + run), False, []
                while not done:
                    obs, _, done, out = env.step(policy(obs, env.twin.state, env.mask()))
                    lines.append(out)
                first[lines[0]["regimen"]] += 1
                alive.append(sum(line["months"] for line in lines))
        results[name] = {"survival_months_5y": float(np.mean(alive)),
                         "most_common_first_choice": first.most_common(1)[0][0]}
    return results


def kaplan_meier(times: np.ndarray, events: np.ndarray, at: list[float]) -> list[float]:
    survival, curve, at_risk = 1.0, [], len(times)
    for t in np.unique(times):
        here = times == t
        survival *= 1.0 - events[here].sum() / at_risk
        at_risk -= here.sum()
        curve.append((t, survival))
    return [float(next((s for t, s in reversed(curve) if t <= m), 1.0)) for m in at]


def simulator_check(histories: list[dict], twins: list[dict], model: OutcomeModel, regimens: list[list[str]],
                    rollouts: int = 20, seed: int = 7) -> dict:
    """Is the twin a realistic environment? Give each held-out patient's twin the treatments the patient
    really received (then standard care) and compare simulated with real survival from the first line."""
    patients = {p["patient_id"]: p for p in histories}
    chosen = [t for t in twins if t["split"] != "train" and patients[t["patient_id"]]["lines"]]
    months = [12.0, 24.0, 36.0, 48.0, 60.0]
    first_lines = [patients[t["patient_id"]]["lines"][0]["outcome"] for t in chosen]
    real = kaplan_meier(np.array([o["os_months"] for o in first_lines]),
                        np.array([o["os_event"] for o in first_lines]), months)
    care = standard_care(regimens)
    deaths = []
    for k, record in enumerate(chosen):
        given = [line["regimen"] for line in patients[record["patient_id"]]["lines"]]
        for run in range(rollouts):
            rng = random.Random(seed + 1000 * k + run)
            twin = DigitalTwin(record["state"], model, record["multipliers"])
            while not twin.ended and twin.state["line_number"] <= 12:
                n = twin.state["line_number"] - record["state"]["line_number"]
                regimen = given[n] if n < len(given) else regimens[care(None, twin.state, allowed(twin.state, regimens))]
                twin.step(regimen, rng)
            deaths.append(twin.months_elapsed if twin.ended == "death" else math.inf)
    simulated = [float(np.mean([d > m for d in deaths])) for m in months]
    return {"patients": len(chosen), "months": months, "real_alive": real, "twin_alive": simulated}
