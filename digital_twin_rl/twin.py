"""Per-patient digital twin.

A twin is the shared outcome-model ensemble instantiated at one patient's
real state (stage, grade, residual tumor, age, line, disease setting, prior
exposures, ...) plus the knowledge-based side-effect model. When the twin
starts after some observed lines, the patient's own history recalibrates it
(shrunken logit offsets), so two patients in the same state can still have
different twins.

    twin.predict(regimen)  -> expected response, progression, survival and
                              side effects for that regimen, with uncertainty
    twin.step(regimen)     -> one sampled treatment line; the twin's state
                              moves to the next decision (or death/remission)
"""
from __future__ import annotations

import copy
import math
import random

import numpy as np

from .outcome_model import (HORIZON_MONTHS, TIME_BINS, OutcomeEnsemble, months_bucket, restricted_mean_months,
                            survival_curve, time_bin)
from .toxicity import ToxicityModel

LINE_DURATION_MONTHS = 4.0     # about six 3-weekly cycles; used to age platinum-free intervals
CALIBRATION_PRIOR_LINES = 5.0  # shrinkage strength for patient-specific offsets
# The open-ended last bin is closed at the horizon, exactly as restricted_mean_months does.
LAST_BIN_WIDTH = HORIZON_MONTHS - TIME_BINS[-1]


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _shift(p: np.ndarray, offset: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(p) + offset))) if offset else p


def _sample_category(distribution: dict[str, float], rng: random.Random) -> str | None:
    if not distribution:
        return None
    r, cumulative = rng.random(), 0.0
    for key, p in distribution.items():
        cumulative += p
        if r < cumulative:
            return key
    return key


def hazard_shift_for_rate(hazards: np.ndarray, target: float, months: float = 12.0) -> float:
    """Logit shift of every bin hazard that makes the mean event rate by `months` equal `target`."""
    index = time_bin(months) - 1
    low, high = -6.0, 6.0
    for _ in range(40):
        mid = (low + high) / 2.0
        rate = float(np.mean(1.0 - survival_curve(_shift(hazards, mid))[:, index]))
        low, high = (mid, high) if rate < target else (low, mid)
    return (low + high) / 2.0


def initial_state(static: dict) -> dict:
    """State for a newly diagnosed patient with no recorded treatment lines."""
    return {
        "age_years": static.get("age_at_diagnosis"), "stage": static.get("stage"),
        "grade": static.get("grade"), "residual": static.get("residual"),
        "line_number": 1, "months_since_diagnosis": 1.0, "setting": "primary",
        "last_tumor_status": None, "ecog": None, "karnofsky": None, "prior_exposure": {},
        "months_since_last_platinum": None, "months_since_last_line": None,
        "undated_history": static.get("undated_history", 0),
    }


def sample_event_time(hazards: np.ndarray, u: float) -> float | None:
    """Inverse-CDF sample from discrete-time hazards; None if no event in modelled range."""
    survival_before = 1.0
    for j, h in enumerate(hazards):
        survival_after = survival_before * (1.0 - h)
        if survival_after < u:
            width = (TIME_BINS[j + 1] - TIME_BINS[j]) if j + 1 < len(TIME_BINS) else LAST_BIN_WIDTH
            fraction = (survival_before - u) / max(survival_before - survival_after, 1e-9)
            return TIME_BINS[j] + width * fraction
        survival_before = survival_after
    return None


class PatientDigitalTwin:
    def __init__(self, patient: dict, ensemble: OutcomeEnsemble, toxicity: ToxicityModel,
                 start_line: int = 0, members: list[int] | None = None, calibrate: bool = True):
        self.patient_id = patient["patient_id"]
        self.split = patient["split"]
        self.ensemble = ensemble
        self.toxicity = toxicity
        self.members = members if members is not None else list(range(len(ensemble)))
        lines = patient["lines"]
        if lines:
            if not 0 <= start_line < len(lines):
                raise ValueError(f"start_line {start_line} is outside the patient's {len(lines)} observed lines")
            self.state = copy.deepcopy(lines[start_line]["state"])
            self.state_source = f"observed line {start_line + 1}"
        else:
            start_line = 0
            self.state = initial_state(patient["static"])
            self.state_source = "synthesized from diagnosis record (no dated treatment lines)"
        self.state["severe_toxicity_history"] = {}
        self.observed_before = lines[:start_line]
        self.offsets = self._calibrate() if calibrate else {"response": 0.0, "pfs": 0.0}
        self.months_elapsed = 0.0
        self.end_reason: str | None = None   # "death", "long_term_remission" or "horizon"

    def _calibrate(self) -> dict[str, float]:
        """Shrunken offsets from the patient's own earlier observed lines.

        Only response and progression are used: both are observed before the
        current line starts. Overall survival of earlier lines would reveal
        follow-up after the current decision, so it is not used.
        """
        offsets = {"response": 0.0, "pfs": 0.0}
        if not self.observed_before:
            return offsets
        pred = self.ensemble.predict([l["state"] for l in self.observed_before],
                                     [l["regimen"] for l in self.observed_before], self.members)
        responses = [(l["outcome"]["response"], p) for l, p in zip(self.observed_before, pred["response"].mean(0))
                     if l["outcome"]["response"] is not None]
        if responses:
            observed = np.clip(np.mean([o for o, _ in responses]), 0.05, 0.95)
            predicted = np.clip(np.mean([p for _, p in responses]), 0.05, 0.95)
            weight = len(responses) / (len(responses) + CALIBRATION_PRIOR_LINES)
            offsets["response"] = float(weight * (_logit(observed) - _logit(predicted)))
        known = [i for i, l in enumerate(self.observed_before)
                 if l["outcome"]["pfs_event"] or l["outcome"]["pfs_months"] >= 12]
        if known:
            observed = np.clip(np.mean([int(self.observed_before[i]["outcome"]["pfs_event"]
                                            and self.observed_before[i]["outcome"]["pfs_months"] < 12) for i in known]),
                               0.05, 0.95)
            weight = len(known) / (len(known) + CALIBRATION_PRIOR_LINES)
            offsets["pfs"] = float(weight * hazard_shift_for_rate(pred["pfs"].mean(0)[known], observed))
        return offsets

    def _member_predictions(self, regimen: list[str], members: list[int]) -> dict[str, np.ndarray]:
        pred = self.ensemble.predict([self.state], [regimen], members)
        return {
            "response": _shift(pred["response"][:, 0], self.offsets["response"]),
            "pfs": _shift(pred["pfs"][:, 0, :], self.offsets["pfs"]),
            "death_cause": pred["death_cause"][:, 0, :],
            "os": pred["os"][:, 0, :],
        }

    @staticmethod
    def _death_share(pfs_hazards: np.ndarray, death_cause: np.ndarray) -> np.ndarray:
        """Probability that the first event (within the modelled range) is death."""
        before = np.concatenate([np.ones(pfs_hazards.shape[:-1] + (1,)), survival_curve(pfs_hazards)[..., :-1]], axis=-1)
        mass = before * pfs_hazards
        return (mass * death_cause).sum(-1) / np.maximum(mass.sum(-1), 1e-9)

    def predict(self, regimen: list[str]) -> dict:
        """Expected outcomes of giving `regimen` now, averaged over ensemble members."""
        pred = self._member_predictions(regimen, self.members)
        pfs_curve = survival_curve(pred["pfs"])
        os_curve = survival_curve(pred["os"])
        pfs_rmst = restricted_mean_months(pred["pfs"])
        os_rmst = restricted_mean_months(pred["os"])
        at = {m: time_bin(m) - 1 for m in (6.0, 12.0, 24.0)}
        return {
            "regimen": "+".join(regimen),
            "response_probability": float(pred["response"].mean()),
            "progression_free_probability": {f"{int(m)}m": float(pfs_curve[:, i].mean()) for m, i in at.items()},
            "expected_progression_free_months_5y": float(pfs_rmst.mean()),
            "probability_first_event_is_death": float(self._death_share(pred["pfs"], pred["death_cause"]).mean()),
            "survival_probability": {f"{int(m)}m": float(os_curve[:, i].mean()) for m, i in at.items()},
            "expected_survival_months_5y": float(os_rmst.mean()),
            "side_effects": self.toxicity.risk(self.state, regimen),
            "uncertainty": {
                "response_std": float(pred["response"].std()),
                "progression_free_months_std": float(pfs_rmst.std()),
                "survival_months_std": float(os_rmst.std()),
            },
        }

    def step(self, regimen: list[str], rng: random.Random, member: int | None = None) -> dict:
        """Simulate one treatment line and advance the twin to its next decision point."""
        if self.end_reason is not None:
            raise RuntimeError(f"Twin has no further decision point ({self.end_reason})")
        chosen = member if member is not None else rng.choice(self.members)
        pred = self._member_predictions(regimen, [chosen])
        all_pred = self._member_predictions(regimen, self.members)
        events, tox = self.toxicity.sample(self.state, regimen, rng)
        # One shared uniform couples response and the first event: responders
        # tend to stay progression-free longer, as observed clinically.
        u = rng.random()
        response = int(u < float(pred["response"][0]))
        # Competing risks: the progression-free model gives the time of the
        # first event (progression or death); the cause model splits it.
        t_event = sample_event_time(pred["pfs"][0], u)
        event_is_death = t_event is not None and rng.random() < float(pred["death_cause"][0][time_bin(t_event)])
        duration = t_event if t_event is not None else math.inf
        remaining = HORIZON_MONTHS - self.months_elapsed
        within = duration <= remaining
        months_alive = min(duration, remaining)
        if t_event is None:
            self.end_reason = "long_term_remission"
        elif not within:
            self.end_reason = "horizon"
        else:
            self.end_reason = "death" if event_is_death else None
        progressed = t_event is not None and within and not event_is_death

        outcome = {
            "regimen": "+".join(regimen), "member": chosen, "response": response,
            "severe_side_effects": events, "side_effect_risk": tox,
            "months_on_line": months_alive, "died": self.end_reason == "death",
            "progressed": progressed, "end_reason": self.end_reason,
            # Expectations under the chosen member, for low-variance training rewards.
            "expected": {"months_on_line": float(restricted_mean_months(pred["pfs"][0], remaining)),
                         "response": float(pred["response"][0]),
                         "severe_side_effects": float(sum(tox["severe_event_probability"].values()))},
            "uncertainty": {"response_std": float(all_pred["response"].std()),
                            "survival_months_std": float(restricted_mean_months(all_pred["os"]).std()),
                            "progression_free_months_std": float(restricted_mean_months(all_pred["pfs"]).std())},
        }
        self.months_elapsed += months_alive
        self._advance(regimen, outcome, months_alive, rng)
        return outcome

    def _advance(self, regimen: list[str], outcome: dict, months_on_line: float, rng: random.Random) -> None:
        s = self.state
        history = s.setdefault("severe_toxicity_history", {})
        for organ in outcome["severe_side_effects"]:
            history[organ] = history.get(organ, 0) + 1
        # ECOG is never recorded at line start in this export, so the outcome
        # model has not seen it; performance decline after severe side effects
        # is tracked separately and used only by the side-effect and safety rules.
        if outcome["severe_side_effects"]:
            s["performance_decline"] = s.get("performance_decline", 0) + 1
        if not outcome["progressed"]:
            return
        if s.get("age_years") is not None:
            s["age_years"] += months_on_line / 12.0
        s["months_since_diagnosis"] = s.get("months_since_diagnosis", 0.0) + months_on_line
        exposure = s.setdefault("prior_exposure", {})
        for c in regimen:
            exposure[c] = exposure.get(c, 0) + 1
        transitions = self.ensemble.transitions
        setting = status = None
        if transitions:
            # Record the next state the way the data records it: intervals run
            # from the line's end date when one exists, otherwise from its start;
            # setting and last tumor status follow the training-line frequencies.
            if rng.random() < transitions["end_day_known_rate"]:
                quantiles = transitions["treatment_months_quantiles"]
                on_treatment = min(months_on_line, quantiles[rng.randrange(len(quantiles))])
            else:
                on_treatment = 0.0
            setting = _sample_category(transitions["next_setting_by_months_to_event"][months_bucket(months_on_line)], rng)
            status = _sample_category(transitions["next_tumor_status_by_response"].get(str(outcome["response"]), {}), rng)
        else:
            on_treatment = min(months_on_line, LINE_DURATION_MONTHS)
        if setting is None:
            setting = "progressive" if months_on_line < 6.0 else "recurrent"
        off_treatment = max(0.0, months_on_line - on_treatment)
        if "platinum" in regimen:
            s["months_since_last_platinum"] = off_treatment
        elif s.get("months_since_last_platinum") is not None:
            s["months_since_last_platinum"] += months_on_line
        s["months_since_last_line"] = off_treatment
        s["line_number"] = s.get("line_number", 1) + 1
        s["setting"] = setting
        s["last_tumor_status"] = None if status in (None, "None") else status
