"""Step 3: a digital twin for each patient, built from the 5 most similar training patients.

A twin is the outcome model evaluated at this patient's own state, adjusted by what really
happened to the 5 most similar training patients (nearest in the encoded state, one decision
point per patient). For each of the 5 we count the progression and death events that really
happened and the number the model expected over that patient's follow-up. The expected
counts come only from networks that never trained on that patient (out-of-bag), so they show
the model's real error for patients like this one. If they did worse than expected, the twin's
risk goes up; if better, it goes down:

    risk multiplier = (observed events + 20) / (expected events + 20)

The +20 pseudo-events pull the multiplier toward 1, because 5 patients are few. The strength was
chosen on validation patients; `check_twins` measures the effect on held-out patients.

    twin.predict(regimen)    expected response, progression-free and overall survival
    twin.step(regimen, rng)  simulate one treatment line and move to the next decision
"""
from __future__ import annotations

import copy
import random

import numpy as np

from .data import DAYS_PER_MONTH, pre_line_state
from .model import (HORIZON, TIME_BINS, OutcomeModel, bootstrap, duration_bucket, encode_state, expected_months,
                    survival, time_bin, training_patients)

NEIGHBOURS = 5
PRIOR_EVENTS = 20.0
MAX_HAZARD = 0.99


class NeighbourIndex:
    """Every treatment decision of the training patients, searchable by similarity."""

    def __init__(self, histories: list[dict], model: OutcomeModel):
        patients = training_patients(histories)
        bags = [set(bootstrap(patients, seed)) for seed in model.seeds]
        self.entries = []
        for p in histories:
            if p["patient_id"] not in patients:
                continue
            out_of_bag = [m for m, bag in enumerate(bags) if p["patient_id"] not in bag]
            if not out_of_bag:
                continue
            lines = p["lines"]
            pred = model.predict([l["state"] for l in lines], [l["regimen"] for l in lines], out_of_bag)
            for i, line in enumerate(lines):
                o = line["outcome"]
                self.entries.append({"patient_id": p["patient_id"], "state": line["state"],
                                     "pfs": (o["pfs_months"], o["pfs_event"], pred["pfs"][:, i].mean(0)),
                                     "os": (o["os_months"], o["os_event"], pred["os"][:, i].mean(0))})
        vectors = np.array([encode_state(e["state"]) for e in self.entries])
        self.mean, self.scale = vectors.mean(0), vectors.std(0)
        self.scale[self.scale < 1e-9] = 1.0
        self.matrix = (vectors - self.mean) / self.scale

    def nearest(self, state: dict, k: int = NEIGHBOURS, exclude: str | None = None) -> list[dict]:
        """The k most similar decision points, each from a different patient (never `exclude`)."""
        z = (np.array(encode_state(state)) - self.mean) / self.scale
        distances = np.sqrt(((self.matrix - z) ** 2).mean(1))
        chosen, seen = [], {exclude}
        for i in np.argsort(distances, kind="stable"):
            entry = self.entries[i]
            if entry["patient_id"] not in seen:
                seen.add(entry["patient_id"])
                chosen.append({**entry, "similarity": float(1.0 / (1.0 + distances[i]))})
                if len(chosen) == k:
                    break
        return chosen


def risk_multiplier(neighbours: list[dict], outcome: str) -> float:
    """(observed + prior) / (expected + prior) events among the neighbours, for 'pfs' or 'os'.

    Expected events are counted the way the model was trained: the risk of every bin the patient
    lived through, plus the bin where the event happened.
    """
    followed, events, hazards = (np.array(x) for x in zip(*(n[outcome] for n in neighbours)))
    bins = np.array([time_bin(m) for m in followed])[:, None]
    index = np.arange(len(TIME_BINS))[None, :]
    at_risk = (index < bins) | ((index == bins) & (events[:, None] > 0))
    expected = float((hazards * at_risk).sum())
    return float((events.sum() + PRIOR_EVENTS) / (expected + PRIOR_EVENTS))


def event_time(hazards: np.ndarray, u: float) -> float | None:
    """Months until the event for a uniform draw u (inverse CDF); None if no event within the model's range."""
    before = 1.0
    for j, h in enumerate(hazards):
        after = before * (1.0 - h)
        if after < u:
            width = (TIME_BINS[j + 1] if j + 1 < len(TIME_BINS) else HORIZON) - TIME_BINS[j]
            return TIME_BINS[j] + width * (before - u) / max(before - after, 1e-9)
        before = after
    return None


class DigitalTwin:
    def __init__(self, state: dict, model: OutcomeModel, multipliers: dict[str, float],
                 neighbours: list[dict] | None = None):
        self.state = copy.deepcopy(state)
        self.state.setdefault("prior_exposure", {})
        self.model, self.multipliers, self.neighbours = model, multipliers, neighbours or []
        self.months_elapsed = 0.0
        self.ended: str | None = None        # "death", "remission" or "horizon" (5 years reached)

    def _predict(self, regimen: list[str]) -> dict:
        p = {k: v[:, 0] for k, v in self.model.predict([self.state], [regimen]).items()}
        for k in ("pfs", "os"):
            p[k] = np.minimum(p[k] * self.multipliers[k], MAX_HAZARD)
        return p

    def predict(self, regimen: list[str]) -> dict:
        """Expected effect of giving `regimen` now (mean over the 10 networks, and their spread)."""
        p = self._predict(regimen)
        pfs_months, os_months = expected_months(p["pfs"]), expected_months(p["os"])
        at_12, at_24 = time_bin(12.0) - 1, time_bin(24.0) - 1
        return {
            "response": float(p["response"].mean()), "response_std": float(p["response"].std()),
            "progression_free_months": float(pfs_months.mean()), "progression_free_months_std": float(pfs_months.std()),
            "survival_months": float(os_months.mean()), "survival_months_std": float(os_months.std()),
            "progression_free_12m": float(survival(p["pfs"])[:, at_12].mean()),
            "survival_12m": float(survival(p["os"])[:, at_12].mean()),
            "survival_24m": float(survival(p["os"])[:, at_24].mean()),
        }

    def step(self, regimen: list[str], rng: random.Random) -> dict:
        """Simulate one treatment line with one randomly chosen network, then move to the next decision."""
        if self.ended:
            raise RuntimeError(f"The twin has no further decisions ({self.ended})")
        p = self._predict(regimen)
        m = rng.randrange(len(self.model))
        remaining = HORIZON - self.months_elapsed
        u = rng.random()                  # one shared draw: responders tend to stay progression-free longer
        response = int(u < p["response"][m])
        t = event_time(p["pfs"][m], u)
        if t is None or t >= remaining:
            months, self.ended = remaining, "remission" if t is None else "horizon"
        else:
            months = t
            if rng.random() < p["death"][m][time_bin(t)]:
                self.ended = "death"
        self.months_elapsed += months
        if not self.ended:
            self._next_line(regimen, months, rng)
        return {"regimen": "+".join(regimen), "months": months, "response": response, "ended": self.ended,
                "expected_months": float(expected_months(p["pfs"][m], remaining)),
                "expected_response": float(p["response"][m]),
                "uncertainty": float(p["response"].std() + expected_months(p["os"]).std() / 12.0)}

    def _next_line(self, regimen: list[str], months: float, rng: random.Random) -> None:
        """The cancer progressed: build the state before the next line, recorded as the dataset records it."""
        s, t = self.state, self.model.transitions
        if s.get("age_years") is not None:
            s["age_years"] += months / 12.0
        s["months_since_diagnosis"] += months
        for c in regimen:
            s["prior_exposure"][c] = s["prior_exposure"].get(c, 0) + 1
        # Intervals run from the line's end date when the data would have one, otherwise from its start.
        on_treatment = min(months, rng.choice(t["treatment_months"])) if rng.random() < t["end_date_known"] else 0.0
        if "platinum" in regimen:
            s["months_since_last_platinum"] = months - on_treatment
        elif s.get("months_since_last_platinum") is not None:
            s["months_since_last_platinum"] += months
        s["months_since_last_line"] = months - on_treatment
        s["line_number"] += 1
        settings = t["next_setting"][duration_bucket(months)]
        s["setting"] = rng.choices(list(settings), weights=list(settings.values()))[0]


def build_twin(state: dict, model: OutcomeModel, index: NeighbourIndex, exclude: str | None = None) -> DigitalTwin:
    """A new twin for a patient in `state`, personalised by the 5 most similar training patients."""
    neighbours = index.nearest(state, NEIGHBOURS, exclude)
    multipliers = {k: risk_multiplier(neighbours, k) for k in ("pfs", "os")}
    return DigitalTwin(state, model, multipliers, neighbours)


def starting_state(patient: dict) -> dict:
    """The state at the patient's first treatment decision (one month after diagnosis if none is recorded)."""
    if patient["lines"]:
        return patient["lines"][0]["state"]
    return pre_line_state(patient["static"], [], round(DAYS_PER_MONTH), "primary")


def check_twins(histories: list[dict], model: OutcomeModel, index: NeighbourIndex, split: str = "test") -> dict:
    """Brier score (lower is better) of the 12-month chance of progression and of death, at every treatment
    decision of held-out patients: the shared model alone vs twins built from the 5 most similar patients."""
    from sklearn.metrics import brier_score_loss

    at_12 = time_bin(12.0) - 1
    scores = {}
    for kind in ("pfs", "os"):
        labels, shared, twin = [], [], []
        for p in histories:
            for line in p["lines"] if p["split"] == split else []:
                months, event = line["outcome"][f"{kind}_months"], line["outcome"][f"{kind}_event"]
                if months < 12.0 and not event:
                    continue                     # followed for under a year without an event: unknown
                hazards = model.predict([line["state"]], [line["regimen"]])[kind][:, 0]
                multiplier = risk_multiplier(index.nearest(line["state"]), kind)
                labels.append(int(event and months < 12.0))
                shared.append(1.0 - survival(hazards)[:, at_12].mean())
                twin.append(1.0 - survival(np.minimum(hazards * multiplier, MAX_HAZARD))[:, at_12].mean())
        name = "progression_within_12m" if kind == "pfs" else "death_within_12m"
        scores[name] = {"decisions": len(labels), "brier_shared_model": round(brier_score_loss(labels, shared), 4),
                        "brier_5_neighbour_twin": round(brier_score_loss(labels, twin), 4)}
    return scores


def build_all_twins(histories: list[dict], model: OutcomeModel, index: NeighbourIndex) -> list[dict]:
    """One twin per patient, as JSON records; a patient is never their own neighbour."""
    records = []
    for patient in histories:
        twin = build_twin(starting_state(patient), model, index, exclude=patient["patient_id"])
        records.append({"patient_id": patient["patient_id"], "split": patient["split"], "state": twin.state,
                        "multipliers": twin.multipliers,
                        "neighbours": [{"patient_id": n["patient_id"], "similarity": round(n["similarity"], 3)}
                                       for n in twin.neighbours]})
    return records
