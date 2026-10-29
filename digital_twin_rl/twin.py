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

from .outcome_model import (HORIZON_MONTHS, TIME_BINS, OutcomeEnsemble, restricted_mean_months,
                            survival_curve, time_bin)
from .toxicity import ToxicityModel

LINE_DURATION_MONTHS = 4.0     # about six 3-weekly cycles; used to age platinum-free intervals
CALIBRATION_PRIOR_LINES = 5.0  # shrinkage strength for patient-specific offsets
LAST_BIN_WIDTH = 24.0          # width used to place events inside the open-ended last bin


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _shift(p: np.ndarray, offset: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(p) + offset))) if offset else p


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
            start_line = min(start_line, len(lines) - 1)
            self.state = copy.deepcopy(lines[start_line]["state"])
            self.state_source = f"observed line {start_line + 1}"
        else:
            start_line = 0
            self.state = initial_state(patient["static"])
            self.state_source = "synthesized from diagnosis record (no dated treatment lines)"
        self.state["severe_toxicity_history"] = {}
        self.observed_before = lines[:start_line]
        self.offsets = self._calibrate() if calibrate else {"response": 0.0, "pfs": 0.0, "os": 0.0}
        self.months_elapsed = 0.0
        self.end_reason: str | None = None   # "death", "long_term_remission" or "horizon"

    def _calibrate(self) -> dict[str, float]:
        """Shrunken logit offsets from the patient's own earlier observed lines."""
        offsets = {"response": 0.0, "pfs": 0.0, "os": 0.0}
        if not self.observed_before:
            return offsets
        pred = self.ensemble.predict([l["state"] for l in self.observed_before],
                                     [l["regimen"] for l in self.observed_before], self.members)
        s12 = time_bin(12.0) - 1
        targets = {
            "response": [(l["outcome"]["response"], p) for l, p in zip(self.observed_before, pred["response"].mean(0))
                         if l["outcome"]["response"] is not None],
            "pfs": [(int(l["outcome"]["pfs_months"] < 12 and l["outcome"]["pfs_event"]), 1 - s)
                    for l, s in zip(self.observed_before, survival_curve(pred["pfs"].mean(0))[:, s12])
                    if l["outcome"]["pfs_event"] or l["outcome"]["pfs_months"] >= 12],
            "os": [(int(l["outcome"]["os_months"] < 12 and l["outcome"]["os_event"]), 1 - s)
                   for l, s in zip(self.observed_before, survival_curve(pred["os"].mean(0))[:, s12])
                   if l["outcome"]["os_event"] or l["outcome"]["os_months"] >= 12],
        }
        for key, pairs in targets.items():
            if not pairs:
                continue
            observed = np.clip(np.mean([o for o, _ in pairs]), 0.05, 0.95)
            predicted = np.clip(np.mean([p for _, p in pairs]), 0.05, 0.95)
            weight = len(pairs) / (len(pairs) + CALIBRATION_PRIOR_LINES)
            offsets[key] = float(weight * (_logit(observed) - _logit(predicted)))
        return offsets

    def _member_predictions(self, regimen: list[str], members: list[int]) -> dict[str, np.ndarray]:
        pred = self.ensemble.predict([self.state], [regimen], members)
        return {
            "response": _shift(pred["response"][:, 0], self.offsets["response"]),
            "pfs": _shift(pred["pfs"][:, 0, :], self.offsets["pfs"]),
            "os": _shift(pred["os"][:, 0, :], self.offsets["os"]),
        }

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
        # One shared uniform couples response and progression: responders tend
        # to have longer progression-free intervals, as observed clinically.
        u = rng.random()
        response = int(u < float(pred["response"][0]))
        t_progression = sample_event_time(pred["pfs"][0], u)
        t_death = sample_event_time(pred["os"][0], rng.random())

        if t_death is not None and (t_progression is None or t_death <= t_progression):
            duration, died, progressed = t_death, True, False
        elif t_progression is not None:
            duration, died, progressed = t_progression, False, True
        else:
            duration, died, progressed = math.inf, False, False
        remaining = HORIZON_MONTHS - self.months_elapsed
        within = duration <= remaining
        months_alive = min(duration, remaining)
        if died and within:
            self.end_reason = "death"
        elif progressed and within:
            self.end_reason = None
        elif not died and not progressed:
            self.end_reason = "long_term_remission"
        else:
            self.end_reason = "horizon"

        outcome = {
            "regimen": "+".join(regimen), "member": chosen, "response": response,
            "severe_side_effects": events, "side_effect_risk": tox,
            "months_on_line": months_alive, "died": self.end_reason == "death",
            "progressed": progressed and within, "end_reason": self.end_reason,
            # Expectations under the chosen member, for low-variance training rewards.
            "expected": {"months_on_line": float(restricted_mean_months(pred["pfs"][0], remaining)),
                         "response": float(pred["response"][0]),
                         "severe_side_effects": float(sum(tox["severe_event_probability"].values()))},
            "uncertainty": {"response_std": float(all_pred["response"].std()),
                            "survival_months_std": float(restricted_mean_months(all_pred["os"]).std()),
                            "progression_free_months_std": float(restricted_mean_months(all_pred["pfs"]).std())},
        }
        self.months_elapsed += months_alive
        self._advance(regimen, outcome, months_alive)
        return outcome

    def _advance(self, regimen: list[str], outcome: dict, months_on_line: float) -> None:
        s = self.state
        history = s.setdefault("severe_toxicity_history", {})
        for organ in outcome["severe_side_effects"]:
            history[organ] = history.get(organ, 0) + 1
        if outcome["severe_side_effects"]:
            s["ecog"] = min(4.0, (s["ecog"] if s.get("ecog") is not None else 1.0) + 1.0)
        if not outcome["progressed"]:
            return
        if s.get("age_years") is not None:
            s["age_years"] += months_on_line / 12.0
        s["months_since_diagnosis"] = s.get("months_since_diagnosis", 0.0) + months_on_line
        exposure = s.setdefault("prior_exposure", {})
        for c in regimen:
            exposure[c] = exposure.get(c, 0) + 1
        off_treatment = max(0.0, months_on_line - LINE_DURATION_MONTHS)
        if "platinum" in regimen:
            s["months_since_last_platinum"] = off_treatment
        elif s.get("months_since_last_platinum") is not None:
            s["months_since_last_platinum"] += months_on_line
        s["months_since_last_line"] = off_treatment
        s["line_number"] = s.get("line_number", 1) + 1
        s["setting"] = "recurrent" if outcome["response"] else "progressive"
        s["last_tumor_status"] = "WT"
