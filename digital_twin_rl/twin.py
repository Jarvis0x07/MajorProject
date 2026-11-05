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

NEIGHBOURS = 5        # how many similar patients each twin is built from
PRIOR_EVENTS = 20.0   # how strongly the risk multiplier is pulled toward 1 (no change)
MAX_HAZARD = 0.99     # a risk per time bin can never reach 100% after multiplying


# ---------------------------------------------------------------------------------------------
# Finding similar patients.
# ---------------------------------------------------------------------------------------------
class NeighbourIndex:
    """Every treatment decision of the training patients, searchable by similarity."""

    def __init__(self, histories: list[dict], model: OutcomeModel):
        # Work out which training patients each of the 10 networks trained on (its bootstrap sample).
        patients = training_patients(histories)
        bags = [set(bootstrap(patients, seed)) for seed in model.seeds]
        self.entries = []
        for p in histories:
            if p["patient_id"] not in patients:
                continue
            # Networks that never saw this patient ("out-of-bag"). Skip the rare patient seen by all 10.
            out_of_bag = [m for m, bag in enumerate(bags) if p["patient_id"] not in bag]
            if not out_of_bag:
                continue
            # Store every treatment line of the patient: its state, what really happened (months
            # followed, event yes/no) and the out-of-bag networks' predicted risks per time bin.
            lines = p["lines"]
            pred = model.predict([l["state"] for l in lines], [l["regimen"] for l in lines], out_of_bag)
            for i, line in enumerate(lines):
                o = line["outcome"]
                self.entries.append({"patient_id": p["patient_id"], "state": line["state"],
                                     "pfs": (o["pfs_months"], o["pfs_event"], pred["pfs"][:, i].mean(0)),
                                     "os": (o["os_months"], o["os_event"], pred["os"][:, i].mean(0))})
        # Standardise every feature (mean 0, spread 1) so each one counts equally in the distance.
        vectors = np.array([encode_state(e["state"]) for e in self.entries])
        self.mean, self.scale = vectors.mean(0), vectors.std(0)
        self.scale[self.scale < 1e-9] = 1.0
        self.matrix = (vectors - self.mean) / self.scale

    def nearest(self, state: dict, k: int = NEIGHBOURS, exclude: str | None = None) -> list[dict]:
        """The k most similar decision points, each from a different patient (never `exclude`)."""
        # Distance from this state to every stored decision point (root-mean-square over features).
        z = (np.array(encode_state(state)) - self.mean) / self.scale
        distances = np.sqrt(((self.matrix - z) ** 2).mean(1))
        # Take the closest ones, at most one per patient, until we have k.
        chosen, seen = [], {exclude}
        for i in np.argsort(distances, kind="stable"):
            entry = self.entries[i]
            if entry["patient_id"] not in seen:
                seen.add(entry["patient_id"])
                # Similarity: 1 when identical, falling toward 0 as the distance grows.
                chosen.append({**entry, "similarity": float(1.0 / (1.0 + distances[i]))})
                if len(chosen) == k:
                    break
        return chosen


def risk_multiplier(neighbours: list[dict], outcome: str) -> float:
    """(observed + prior) / (expected + prior) events among the neighbours, for 'pfs' or 'os'.

    Expected events are counted the way the model was trained: the risk of every bin the patient
    lived through, plus the bin where the event happened.
    """
    # For each neighbour: months followed, whether the event happened, predicted risk per bin.
    followed, events, hazards = (np.array(x) for x in zip(*(n[outcome] for n in neighbours)))
    # Which bins each neighbour was "at risk" in during their follow-up.
    bins = np.array([time_bin(m) for m in followed])[:, None]
    index = np.arange(len(TIME_BINS))[None, :]
    at_risk = (index < bins) | ((index == bins) & (events[:, None] > 0))
    # Expected events = sum of the predicted risks over those bins; compare with real events.
    expected = float((hazards * at_risk).sum())
    return float((events.sum() + PRIOR_EVENTS) / (expected + PRIOR_EVENTS))


# ---------------------------------------------------------------------------------------------
# Simulation helper: draw a random event time from a survival curve.
# ---------------------------------------------------------------------------------------------
def event_time(hazards: np.ndarray, u: float) -> float | None:
    """Months until the event for a uniform draw u (inverse CDF); None if no event within the model's range."""
    # Walk down the survival curve bin by bin; the event happens in the bin where survival
    # drops below u, at a point inside that bin found by straight-line interpolation.
    before = 1.0
    for j, h in enumerate(hazards):
        after = before * (1.0 - h)
        if after < u:
            width = (TIME_BINS[j + 1] if j + 1 < len(TIME_BINS) else HORIZON) - TIME_BINS[j]
            return TIME_BINS[j] + width * (before - u) / max(before - after, 1e-9)
        before = after
    return None


# ---------------------------------------------------------------------------------------------
# The digital twin.
# ---------------------------------------------------------------------------------------------
class DigitalTwin:
    def __init__(self, state: dict, model: OutcomeModel, multipliers: dict[str, float],
                 neighbours: list[dict] | None = None):
        # The twin keeps its own copy of the patient state (simulation changes it), the shared
        # model, the risk multipliers from the 5 similar patients, and the clock.
        self.state = copy.deepcopy(state)
        self.state.setdefault("prior_exposure", {})
        self.model, self.multipliers, self.neighbours = model, multipliers, neighbours or []
        self.months_elapsed = 0.0
        self.ended: str | None = None        # "death", "remission" or "horizon" (5 years reached)

    def _predict(self, regimen: list[str]) -> dict:
        # The shared model's prediction for this state + regimen (one row per network), with the
        # progression and death risks multiplied by this twin's personal multipliers.
        p = {k: v[:, 0] for k, v in self.model.predict([self.state], [regimen]).items()}
        for k in ("pfs", "os"):
            p[k] = np.minimum(p[k] * self.multipliers[k], MAX_HAZARD)
        return p

    def predict(self, regimen: list[str]) -> dict:
        """Expected effect of giving `regimen` now (mean over the 10 networks, and their spread)."""
        # What the website shows for each medication: averages over the 10 networks, and the
        # spread between them (_std) as a measure of uncertainty.
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
        # Predictions from all networks; one of them is picked at random to play out this line,
        # so the simulation includes the model's own uncertainty.
        p = self._predict(regimen)
        m = rng.randrange(len(self.model))
        remaining = HORIZON - self.months_elapsed

        # Response and time to the first event (progression or death) share one random draw.
        u = rng.random()                  # one shared draw: responders tend to stay progression-free longer
        response = int(u < p["response"][m])
        t = event_time(p["pfs"][m], u)

        # Decide how the line ends: no event at all (remission), past the 5-year mark (horizon),
        # or an event that is either death or progression (then the next treatment line follows).
        if t is None or t >= remaining:
            months, self.ended = remaining, "remission" if t is None else "horizon"
        else:
            months = t
            if rng.random() < p["death"][m][time_bin(t)]:
                self.ended = "death"
        self.months_elapsed += months
        if not self.ended:
            self._next_line(regimen, months, rng)

        # Report what happened, plus expected values (used for a steadier RL reward) and how
        # much the 10 networks disagree (used for the uncertainty penalty).
        return {"regimen": "+".join(regimen), "months": months, "response": response, "ended": self.ended,
                "expected_months": float(expected_months(p["pfs"][m], remaining)),
                "expected_response": float(p["response"][m]),
                "uncertainty": float(p["response"].std() + expected_months(p["os"]).std() / 12.0)}

    def _next_line(self, regimen: list[str], months: float, rng: random.Random) -> None:
        """The cancer progressed: build the state before the next line, recorded as the dataset records it."""
        s, t = self.state, self.model.transitions
        # Time passes: the patient is older and further from diagnosis.
        if s.get("age_years") is not None:
            s["age_years"] += months / 12.0
        s["months_since_diagnosis"] += months
        # The drugs just given now count as earlier treatment.
        for c in regimen:
            s["prior_exposure"][c] = s["prior_exposure"].get(c, 0) + 1
        # Intervals run from the line's end date when the data would have one, otherwise from its start.
        on_treatment = min(months, rng.choice(t["treatment_months"])) if rng.random() < t["end_date_known"] else 0.0
        # Update the time since platinum and since the last line.
        if "platinum" in regimen:
            s["months_since_last_platinum"] = months - on_treatment
        elif s.get("months_since_last_platinum") is not None:
            s["months_since_last_platinum"] += months
        s["months_since_last_line"] = months - on_treatment
        # Move to the next line, with a disease setting drawn from how often each one followed
        # a line of this length in the real data.
        s["line_number"] += 1
        settings = t["next_setting"][duration_bucket(months)]
        s["setting"] = rng.choices(list(settings), weights=list(settings.values()))[0]


# ---------------------------------------------------------------------------------------------
# Building twins.
# ---------------------------------------------------------------------------------------------
def build_twin(state: dict, model: OutcomeModel, index: NeighbourIndex, exclude: str | None = None) -> DigitalTwin:
    """A new twin for a patient in `state`, personalised by the 5 most similar training patients."""
    # Find the 5 most similar patients, compute the progression and death multipliers from how
    # they really did, and create the twin.
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
                # The real answer: did progression (or death) happen within 12 months?
                months, event = line["outcome"][f"{kind}_months"], line["outcome"][f"{kind}_event"]
                if months < 12.0 and not event:
                    continue                     # followed for under a year without an event: unknown
                # The prediction with the shared model alone, and with the 5-neighbour multiplier applied.
                hazards = model.predict([line["state"]], [line["regimen"]])[kind][:, 0]
                multiplier = risk_multiplier(index.nearest(line["state"]), kind)
                labels.append(int(event and months < 12.0))
                shared.append(1.0 - survival(hazards)[:, at_12].mean())
                twin.append(1.0 - survival(np.minimum(hazards * multiplier, MAX_HAZARD))[:, at_12].mean())
        # Brier score = average squared gap between predicted chance and what happened (0 or 1).
        name = "progression_within_12m" if kind == "pfs" else "death_within_12m"
        scores[name] = {"decisions": len(labels), "brier_shared_model": round(brier_score_loss(labels, shared), 4),
                        "brier_5_neighbour_twin": round(brier_score_loss(labels, twin), 4)}
    return scores


def build_all_twins(histories: list[dict], model: OutcomeModel, index: NeighbourIndex) -> list[dict]:
    """One twin per patient, as JSON records; a patient is never their own neighbour."""
    # For every patient: build the twin at their first decision and save what defines it
    # (starting state, risk multipliers, and which 5 patients it was built from).
    records = []
    for patient in histories:
        twin = build_twin(starting_state(patient), model, index, exclude=patient["patient_id"])
        records.append({"patient_id": patient["patient_id"], "split": patient["split"], "state": twin.state,
                        "multipliers": twin.multipliers,
                        "neighbours": [{"patient_id": n["patient_id"], "similarity": round(n["similarity"], 3)}
                                       for n in twin.neighbours]})
    return records
