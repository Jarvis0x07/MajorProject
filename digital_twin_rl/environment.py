"""Twin-backed treatment environment: the patient twin is the environment, the agent is the doctor.

Each episode starts from a real patient's observed decision point. At every
step the agent picks a regimen; the twin simulates that treatment line (side
effects, response, progression or death) and moves to the next decision.

Reward per line (RewardConfig):
    years alive during the line (until progression/death, within 5 years)
  + bonus if the tumor responds (next assessment tumor-free)
  - penalty per severe side effect
  - penalty proportional to ensemble disagreement (pessimism against
    exploiting model error, as in uncertainty-penalised model-based RL)

Action masks combine illustrative safety rules with an optional BCQ-style
filter that keeps only regimens clinicians plausibly chose in that state.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .outcome_model import HORIZON_MONTHS, STATE_FEATURES, OutcomeEnsemble, encode_state
from .toxicity import ToxicityModel
from .twin import PatientDigitalTwin

MAX_LINES = 8
START_MODES = ("first", "relapse", "random")


@dataclass
class RewardConfig:
    per_year_alive: float = 1.0
    response_bonus: float = 0.25
    severe_side_effect_penalty: float = 0.15
    death_penalty: float = 0.0
    uncertainty_penalty: float = 0.5
    # Use the twin's expected months/response/side effects instead of the sampled
    # ones. Transitions stay stochastic; this only lowers reward variance.
    expected: bool = False


def supported_regimens(histories: list[dict], min_support: int = 10) -> list[tuple[list[str], int]]:
    """Regimens given at least `min_support` times in training lines (excluding 'other')."""
    counts = Counter(tuple(line["regimen"]) for p in histories if p["split"] == "train" for line in p["lines"])
    return [(list(r), n) for r, n in counts.most_common() if n >= min_support and "other" not in r]


def safety_mask(state: dict, regimens: list[list[str]]) -> np.ndarray:
    """Illustrative hard constraints; they are simulation rules, not clinical guidance."""
    severe = state.get("severe_toxicity_history") or {}
    exposure = state.get("prior_exposure") or {}
    ecog = state.get("ecog")
    allowed = []
    for regimen in regimens:
        ok = True
        if "anthracyclines" in regimen and (exposure.get("anthracyclines", 0) >= 2 or severe.get("cardiac")):
            ok = False      # cumulative anthracycline cardiotoxicity
        if severe.get("neuropathy") and ({"taxanes", "vinca_alkaloids"} & set(regimen)):
            ok = False      # severe neuropathy: avoid further neurotoxic agents
        if severe.get("hypersensitivity") and "platinum" in regimen:
            ok = False      # severe platinum hypersensitivity
        if ecog is not None and ecog >= 3 and len(regimen) > 1:
            ok = False      # poor performance status: single agents only
        allowed.append(ok)
    mask = np.array(allowed, dtype=bool)
    if not mask.any():
        mask = np.array([len(r) == 1 for r in regimens], dtype=bool)
    return mask if mask.any() else np.ones(len(regimens), dtype=bool)


def action_mask(state: dict, regimens: list[list[str]], behavior=None, bcq_threshold: float = 0.0) -> np.ndarray:
    """Safety mask, narrowed (when possible) to regimens with clinician support >= threshold."""
    mask = safety_mask(state, regimens)
    if bcq_threshold > 0:
        probs = behavior.probabilities(state)
        supported = probs / max(probs.max(), 1e-12) >= bcq_threshold
        if (mask & supported).any():
            mask = mask & supported
    return mask


def build_observation(twin: PatientDigitalTwin, organs: list[str]) -> np.ndarray:
    state = twin.state
    severe = state.get("severe_toxicity_history") or {}
    return np.array(encode_state(state) + [min(severe.get(o, 0), 2) / 2.0 for o in organs]
                    + [twin.months_elapsed / HORIZON_MONTHS], dtype=np.float32)


class TwinTreatmentEnv:
    def __init__(self, histories: list[dict], ensemble: OutcomeEnsemble, toxicity: ToxicityModel,
                 regimens: list[list[str]], behavior=None, split: str = "train",
                 members: list[int] | None = None, reward: RewardConfig | None = None,
                 bcq_threshold: float = 0.0, start: str = "first", max_lines: int = MAX_LINES,
                 calibrate: bool = True, seed: int = 0):
        if start not in START_MODES:
            raise ValueError(f"start must be one of {START_MODES}")
        if bcq_threshold > 0 and behavior is None:
            raise ValueError("A behavior model is required for a BCQ threshold")
        self.ensemble, self.toxicity = ensemble, toxicity
        self.regimens = regimens
        self.behavior = behavior
        self.members = members if members is not None else list(range(len(ensemble)))
        self.reward_config = reward or RewardConfig()
        self.bcq_threshold = bcq_threshold
        self.start, self.max_lines, self.calibrate = start, max_lines, calibrate
        self.rng = random.Random(seed)
        self.organs = toxicity.organs
        self.starts: list[tuple[dict, int]] = []
        for patient in histories:
            if patient["split"] != split or not patient["lines"]:
                continue
            if start == "first":
                self.starts.append((patient, 0))
            elif start == "relapse":
                relapse = next((l["line_index"] for l in patient["lines"]
                                if l["state"]["setting"] in ("recurrent", "progressive")), None)
                if relapse is not None:
                    self.starts.append((patient, relapse))
            else:
                self.starts.extend((patient, i) for i in range(len(patient["lines"])))
        if not self.starts:
            raise ValueError(f"No start points in split {split!r} for start mode {start!r}")
        self.twin: PatientDigitalTwin | None = None
        self.lines_given = 0

    @property
    def observation_dim(self) -> int:
        return len(STATE_FEATURES) + len(self.organs) + 1

    @property
    def action_count(self) -> int:
        return len(self.regimens)

    def observation(self) -> np.ndarray:
        return build_observation(self.twin, self.organs)

    def action_mask(self) -> np.ndarray:
        return action_mask(self.twin.state, self.regimens, self.behavior, self.bcq_threshold)

    def _info(self) -> dict:
        return {"patient_id": self.twin.patient_id, "state": self.twin.state,
                "action_mask": self.action_mask(), "months_elapsed": self.twin.months_elapsed}

    def start_points(self) -> list[tuple[str, int]]:
        return [(p["patient_id"], i) for p, i in self.starts]

    def reset(self, seed: int | None = None, start_index: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng.seed(seed)
        patient, line = self.starts[start_index] if start_index is not None else self.rng.choice(self.starts)
        self.twin = PatientDigitalTwin(patient, self.ensemble, self.toxicity, line, self.members, self.calibrate)
        self.lines_given = 0
        return self.observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.twin is None or self.twin.end_reason is not None:
            raise RuntimeError("Episode is over; call reset()")
        if not self.action_mask()[action]:
            raise ValueError(f"Action {action} ({'+'.join(self.regimens[action])}) is masked in this state")
        outcome = self.twin.step(self.regimens[action], self.rng)
        self.lines_given += 1
        rc = self.reward_config
        uncertainty = outcome["uncertainty"]["response_std"] + outcome["uncertainty"]["survival_months_std"] / 12.0
        source = outcome["expected"] if rc.expected else {
            "months_on_line": outcome["months_on_line"], "response": outcome["response"],
            "severe_side_effects": len(outcome["severe_side_effects"])}
        components = {
            "years_alive": rc.per_year_alive * source["months_on_line"] / 12.0,
            "response": rc.response_bonus * source["response"],
            "side_effects": -rc.severe_side_effect_penalty * source["severe_side_effects"],
            "death": -rc.death_penalty * outcome["died"],
            "uncertainty": -rc.uncertainty_penalty * uncertainty,
        }
        reward = float(sum(components.values()))
        terminated = self.twin.end_reason is not None
        truncated = not terminated and self.lines_given >= self.max_lines
        info = self._info()
        info.update({"outcome": outcome, "reward_components": components})
        return self.observation(), reward, terminated, truncated, info
