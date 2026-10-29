"""Knowledge-based side-effect model for the patient twin.

TCGA-OV records no adverse events, so side effects cannot be learned from this
cohort. This module turns the editable class-level profiles in
toxicity_profiles.json into per-organ probabilities of a severe (grade >= 3)
event for one treatment line, adjusted for age, performance status, prior
exposure, and earlier severe events. Every number here is an assumption.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_PROFILE_PATH = Path(__file__).with_name("toxicity_profiles.json")


class ToxicityModel:
    def __init__(self, profile_path: Path = DEFAULT_PROFILE_PATH):
        config = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        self.organs: list[str] = config["organs"]
        self.class_levels: dict[str, dict[str, int]] = config["class_levels"]
        self.score_to_probability = sorted((float(k), float(v)) for k, v in config["severe_event_probability_by_score"].items())
        self.max_probability = float(config["max_probability"])
        self.overlap_weight = float(config["combination_overlap_weight"])
        self.modifiers = config["modifiers"]
        self.disutility = config["organ_disutility"]

    def _probability(self, score: float) -> float:
        points = self.score_to_probability
        if score <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if score <= x1:
                return y0 + (y1 - y0) * (score - x0) / (x1 - x0)
        return min(self.max_probability, points[-1][1])

    def regimen_scores(self, regimen: list[str]) -> dict[str, float]:
        """Combine class levels: the worst class plus a fraction of the overlap."""
        scores = {}
        for organ in self.organs:
            levels = [self.class_levels.get(c, {}).get(organ, 0) for c in regimen]
            top = max(levels, default=0)
            scores[organ] = top + self.overlap_weight * (sum(levels) - top)
        return scores

    def risk(self, state: dict, regimen: list[str]) -> dict:
        m = self.modifiers
        base = self.regimen_scores(regimen)
        exposure = state.get("prior_exposure") or {}
        severe_history = state.get("severe_toxicity_history") or {}
        age = state.get("age_years")
        ecog = state.get("ecog")
        declined = state.get("performance_decline", 0) >= 1
        probabilities = {}
        for organ, score in base.items():
            if score <= 0:
                probabilities[organ] = 0.0
                continue
            if age is not None and age >= 70:
                score += m["age_at_least_70"]
            if (ecog is not None and ecog >= 2) or declined:
                score += m["ecog_at_least_2"]
            if organ in m["cumulative_organs"]:
                prior_lines = sum(n for cls, n in exposure.items() if self.class_levels.get(cls, {}).get(organ, 0) >= 1)
                score += min(m["cumulative_cap"], m["cumulative_per_prior_line"] * prior_lines)
            if severe_history.get(organ):
                score += m["prior_severe_event_same_organ"]
            probabilities[organ] = self._probability(score)
        any_severe = 1.0
        for p in probabilities.values():
            any_severe *= 1.0 - p
        expected_burden = sum(p * self.disutility.get(o, 1.0) for o, p in probabilities.items())
        return {
            "severe_event_probability": probabilities,
            "levels": {o: level_label(p) for o, p in probabilities.items()},
            "probability_any_severe": 1.0 - any_severe,
            "expected_burden": expected_burden,
        }

    def sample(self, state: dict, regimen: list[str], rng: random.Random) -> tuple[list[str], dict]:
        risk = self.risk(state, regimen)
        events = [o for o, p in risk["severe_event_probability"].items() if p > 0 and rng.random() < p]
        return events, risk


def level_label(probability: float) -> str:
    if probability <= 0:
        return "none"
    if probability < 0.10:
        return "low"
    if probability < 0.25:
        return "moderate"
    return "high"
