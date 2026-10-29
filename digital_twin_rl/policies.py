"""Comparison doctors: random, clinician-imitating (behavior model), and guideline heuristic.

The behavior model estimates P(regimen | state) from observed TCGA-OV lines.
It provides the "what clinicians usually did" baseline and the BCQ-style
support filter. The guideline policy is a simplified, guideline-inspired
heuristic for ovarian cancer (platinum doublet first line and for
platinum-sensitive relapse; single non-platinum agent for platinum-resistant
relapse). It is a benchmark, not clinical advice.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from .outcome_model import encode_state


class BehaviorModel:
    def __init__(self, histories: list[dict], regimens: list[list[str]], c: float = 1.0):
        index = {tuple(r): i for i, r in enumerate(regimens)}
        x, y = [], []
        for patient in histories:
            if patient["split"] != "train":
                continue
            for line in patient["lines"]:
                key = tuple(line["regimen"])
                if key in index:
                    x.append(encode_state(line["state"]))
                    y.append(index[key])
        self.n = len(regimens)
        self.model = LogisticRegression(max_iter=5000, C=c).fit(x, y)
        self.training_rows = len(y)

    def probabilities(self, state: dict) -> np.ndarray:
        probs = np.zeros(self.n)
        probs[self.model.classes_] = self.model.predict_proba([encode_state(state)])[0]
        return probs


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, observation: np.ndarray, info: dict) -> int:
        return self.rng.choice(np.flatnonzero(info["action_mask"]).tolist())


class ClinicianPolicy:
    """Most likely regimen under the behavior model, among allowed actions."""
    name = "clinician"

    def __init__(self, behavior: BehaviorModel):
        self.behavior = behavior

    def act(self, observation: np.ndarray, info: dict) -> int:
        probs = self.behavior.probabilities(info["state"])
        return int(np.argmax(np.where(info["action_mask"], probs, -1.0)))


class GuidelinePolicy:
    name = "guideline"
    RESISTANT_SINGLE_AGENTS = ["anthracyclines", "topoisomerase_inhibitors", "taxanes", "antimetabolites"]

    def __init__(self, regimens: list[list[str]]):
        self.index = {"+".join(r): i for i, r in enumerate(regimens)}

    def _pick(self, mask: np.ndarray, *names: str) -> int | None:
        for name in names:
            i = self.index.get(name)
            if i is not None and mask[i]:
                return i
        return None

    def act(self, observation: np.ndarray, info: dict) -> int:
        state, mask = info["state"], info["action_mask"]
        severe = state.get("severe_toxicity_history") or {}
        exposure = state.get("prior_exposure") or {}
        choice = None
        if state.get("ecog") is not None and state["ecog"] >= 3:
            choice = self._pick(mask, "hormonal_agents")
        pfi = state.get("months_since_last_platinum")
        if choice is None and (state.get("setting") == "primary" or state.get("line_number", 1) == 1):
            choice = self._pick(mask, "platinum+taxanes", "platinum")
        if choice is None and (pfi is None or pfi >= 6.0):
            # Platinum-sensitive (or platinum-naive) relapse: platinum doublet.
            options = (["antimetabolites+platinum", "platinum"] if severe.get("neuropathy")
                       else ["platinum+taxanes", "antimetabolites+platinum", "platinum"])
            choice = self._pick(mask, *options)
        if choice is None:
            # Platinum-resistant relapse: an unused single non-platinum agent first.
            unused = [a for a in self.RESISTANT_SINGLE_AGENTS if not exposure.get(a)]
            choice = self._pick(mask, *unused, *self.RESISTANT_SINGLE_AGENTS, "hormonal_agents")
        return choice if choice is not None else int(np.flatnonzero(mask)[0])
