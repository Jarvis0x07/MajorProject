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

# ---------------------------------------------------------------------------------------------
# Settings for episodes and rewards.
# ---------------------------------------------------------------------------------------------
MAX_LINES = 8               # an episode stops after at most 8 treatment lines
GAMMA = 0.99                # discount: a reward one line later counts 99% as much
RESPONSE_BONUS = 0.25
UNCERTAINTY_PENALTY = 0.5


# ---------------------------------------------------------------------------------------------
# Actions, the guideline constraint, and what the agent observes.
# ---------------------------------------------------------------------------------------------
def supported_regimens(histories: list[dict], min_lines: int = 10) -> list[list[str]]:
    """The action space: regimens given in at least `min_lines` training lines."""
    # Rare regimens are left out: the model has too little data to predict them reliably.
    counts = Counter(tuple(l["regimen"]) for p in histories if p["split"] == "train" for l in p["lines"])
    return [list(r) for r, n in counts.most_common() if n >= min_lines and "other" not in r]


def platinum_resistant(state: dict) -> bool:
    # The cancer came back (not first line) less than 6 months after the last platinum.
    platinum = state.get("months_since_last_platinum")
    return platinum is not None and platinum < 6.0 and state.get("setting") != "primary"


def allowed(state: dict, regimens: list[list[str]]) -> np.ndarray:
    # One True/False per regimen: platinum regimens are blocked for platinum-resistant patients.
    return np.array([not (platinum_resistant(state) and "platinum" in r) for r in regimens])


def observe(state: dict, months_elapsed: float = 0.0) -> np.ndarray:
    # What the agent sees: the encoded patient state plus how far into the 5 years we are.
    return np.array(encode_state(state) + [months_elapsed / HORIZON], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# The environment: one episode = treating one patient's twin.
# ---------------------------------------------------------------------------------------------
class TwinEnv:
    def __init__(self, twins: list[dict], model: OutcomeModel, regimens: list[list[str]], seed: int = 0):
        # `twins` are the saved twin records (from twins.jsonl) this environment can start from.
        self.twins, self.model, self.regimens = twins, model, regimens
        self.rng = random.Random(seed)
        self.twin: DigitalTwin | None = None

    def reset(self, index: int | None = None, seed: int | None = None) -> np.ndarray:
        # Start a new episode: a fresh copy of one twin (a given one, or a random one).
        if seed is not None:
            self.rng.seed(seed)
        record = self.twins[index] if index is not None else self.rng.choice(self.twins)
        self.twin = DigitalTwin(record["state"], self.model, record["multipliers"])
        self.lines = 0
        return observe(self.twin.state, 0.0)

    def mask(self) -> np.ndarray:
        # Which regimens are allowed right now.
        return allowed(self.twin.state, self.regimens)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Give the chosen regimen to the twin and simulate one treatment line.
        out = self.twin.step(self.regimens[action], self.rng)
        self.lines += 1
        # Reward: expected years alive during the line + response bonus - uncertainty penalty.
        reward = (out["expected_months"] / 12.0 + RESPONSE_BONUS * out["expected_response"]
                  - UNCERTAINTY_PENALTY * out["uncertainty"])
        # The episode ends at death, remission, the 5-year mark, or after 8 lines.
        done = out["ended"] is not None or self.lines >= MAX_LINES
        return observe(self.twin.state, self.twin.months_elapsed), reward, done, out


# ---------------------------------------------------------------------------------------------
# The agent: a deep Q-network that scores every regimen for a given state.
# ---------------------------------------------------------------------------------------------
def q_network(n_actions: int) -> nn.Module:
    # Input: the observation (29 state features + time). Output: one Q-value per regimen,
    # i.e. the total future reward the agent expects if it gives that regimen now.
    return nn.Sequential(nn.Linear(len(FEATURES) + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, n_actions))


def best_action(network: nn.Module, observation: np.ndarray, mask: np.ndarray) -> int:
    # The allowed regimen with the highest Q-value (blocked regimens get minus infinity).
    with torch.no_grad():
        q = network(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
    return int(np.argmax(np.where(mask, q, -np.inf)))


def train_agent(env: TwinEnv, episodes: int, seed: int = 42) -> tuple[nn.Module, list[dict]]:
    # Fixed seeds so training gives the same agent every run.
    random.seed(seed)
    torch.manual_seed(seed)
    env.rng.seed(seed)
    batch_rng = np.random.default_rng(seed)
    network = q_network(len(env.regimens))
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-4)
    # Memory of every (observation, action, return) seen so far, used for training batches.
    seen_obs, seen_actions, seen_returns = [], [], []
    history, returns = [], []
    for episode in range(episodes):
        # Exploration rate: start fully random, fall to 5% random over the first 60% of episodes.
        epsilon = max(0.05, 1.0 - 0.95 * episode / (0.6 * episodes))     # explore a lot first, then less

        # Play one episode: sometimes a random allowed regimen (explore), otherwise the best one (exploit).
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

        # Work backwards through the episode: the target for each step is its reward plus the
        # discounted reward of everything that followed (the "return").
        future = 0.0
        for obs_t, action_t, reward_t in reversed(steps):
            future = reward_t + GAMMA * future
            seen_obs.append(obs_t)
            seen_actions.append(action_t)
            seen_returns.append(future)
        returns.append(sum(r for _, _, r in steps))

        # One learning update per step played: pull the Q-values of 128 remembered (state, action)
        # pairs toward their observed returns. Learning starts once 1000 examples are stored.
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

        # Every 1000 episodes, record the average reward per episode (the learning curve).
        if (episode + 1) % 1000 == 0:
            history.append({"episode": episode + 1, "mean_return": float(np.mean(returns[-1000:]))})
            print(f"  episode {episode + 1:>6}  exploring {epsilon:.2f}  mean return {history[-1]['mean_return']:.3f}")
    return network.eval(), history


def save_agent(path: Path, network: nn.Module, regimens: list[list[str]]) -> None:
    # Save the weights with the feature list and the regimens, so they always match on loading.
    torch.save({"features": FEATURES, "regimens": regimens, "weights": network.state_dict()}, path)


def load_agent(path: Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["features"] != FEATURES:
        raise ValueError("The saved agent uses different features; run `python -m digital_twin_rl train-agent`")
    network = q_network(len(checkpoint["regimens"]))
    network.load_state_dict(checkpoint["weights"])
    return network.eval()


# ---------------------------------------------------------------------------------------------
# Evaluation: compare the agent with other "doctors", and check the simulator against reality.
# ---------------------------------------------------------------------------------------------
def standard_care(regimens: list[list[str]]):
    """Baseline doctor: carboplatin + paclitaxel, or liposomal doxorubicin when platinum-resistant."""
    keys = ["+".join(r) for r in regimens]

    def policy(obs: np.ndarray, state: dict, mask: np.ndarray) -> int:
        # The first of these that is allowed; otherwise the first allowed regimen.
        for key in ("platinum+taxanes", "anthracyclines"):
            if key in keys and mask[keys.index(key)]:
                return keys.index(key)
        return int(np.flatnonzero(mask)[0])
    return policy


def compare_policies(env: TwinEnv, network: nn.Module, rollouts: int = 20, seed: int = 7) -> dict:
    """Months alive within 5 years when each doctor treats the twins of the test patients.

    Every doctor sees the same random draws for the same patient and run, so differences are fair.
    """
    # Three doctors: random allowed regimen, standard care, and the trained agent.
    chooser = random.Random(seed)
    policies = {"random": lambda obs, state, mask: chooser.choice(np.flatnonzero(mask).tolist()),
                "standard_care": standard_care(env.regimens),
                "dqn_agent": lambda obs, state, mask: best_action(network, obs, mask)}
    results = {}
    for name, policy in policies.items():
        alive, first = [], Counter()
        # Each doctor treats every test twin 20 times; add up the months alive in each run.
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
    # Standard survival estimate from real data: at each death time, survival drops by the share
    # of patients still being followed who died then. Censored patients just leave the count.
    survival, curve, at_risk = 1.0, [], len(times)
    for t in np.unique(times):
        here = times == t
        survival *= 1.0 - events[here].sum() / at_risk
        at_risk -= here.sum()
        curve.append((t, survival))
    # Read the curve at the requested months.
    return [float(next((s for t, s in reversed(curve) if t <= m), 1.0)) for m in at]


def simulator_check(histories: list[dict], twins: list[dict], model: OutcomeModel, regimens: list[list[str]],
                    rollouts: int = 20, seed: int = 7) -> dict:
    """Is the twin a realistic environment? Give each held-out patient's twin the treatments the patient
    really received (then standard care) and compare simulated with real survival from the first line."""
    # Held-out (validation + test) patients who have recorded treatment lines.
    patients = {p["patient_id"]: p for p in histories}
    chosen = [t for t in twins if t["split"] != "train" and patients[t["patient_id"]]["lines"]]
    months = [12.0, 24.0, 36.0, 48.0, 60.0]

    # Real survival from each patient's first treatment line.
    first_lines = [patients[t["patient_id"]]["lines"][0]["outcome"] for t in chosen]
    real = kaplan_meier(np.array([o["os_months"] for o in first_lines]),
                        np.array([o["os_event"] for o in first_lines]), months)

    # Simulated survival: replay the real regimens line by line (standard care once they run out).
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

    # Share of simulated runs still alive at each month, next to the real share.
    simulated = [float(np.mean([d > m for d in deaths])) for m in months]
    return {"patients": len(chosen), "months": months, "real_alive": real, "twin_alive": simulated}
