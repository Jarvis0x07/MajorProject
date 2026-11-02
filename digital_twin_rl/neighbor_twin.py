"""Create a new patient's twin on the spot from the k most similar stored twins.

The twin database stores one entry per observed treatment line of every
training patient: the encoded pre-line state, the regimen given, what really
happened, and the outcome model's *out-of-bag* prediction for that line (only
ensemble members whose bootstrap sample did not contain that patient). The
out-of-bag error shows how the shared model mis-predicts patients like this one.

For a new patient, the k nearest stored decision points (distinct patients,
z-scored state features) are found. For progression and survival, their
observed events are compared with the events the model expected over each
neighbour's actual follow-up time (observed / expected, using all follow-up,
censored or not), together with the patient's own earlier lines up to today.
The shrunken ratio becomes a hazard offset. Response is not adjusted: only
about one in four lines has a response label, too few among five neighbours.
The result is a new PatientDigitalTwin: the shared model evaluated at this
patient's own state, adjusted by how it mis-predicted the most similar
patients. Every prediction comes from that twin; the neighbours' outcomes are
never returned as answers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .outcome_model import (HORIZON_MONTHS, STATE_FEATURES, TIME_BINS, OutcomeEnsemble, bootstrap_patients,
                            encode_state, survival_curve, time_bin)
from .toxicity import ToxicityModel
from .twin import PatientDigitalTwin, _shift

K_NEIGHBORS = 5
# Expected-event pseudo-counts pulling each observed/expected ratio toward 1, chosen on the validation split
# (twins/neighbor_twin_report.json): neighbours improved 12-month progression; adjusting survival did not help.
PRIOR_EVENTS = {"pfs": 10.0, "os": math.inf}
TWELVE = time_bin(12.0) - 1
DAYS_PER_MONTH = 30.44


def twelve_month_label(months: float, event: int) -> int | None:
    """1 = event before 12 months, 0 = event-free at 12 months, None = censored before 12 months."""
    if event and months < 12.0:
        return 1
    return 0 if months >= 12.0 else None


def line_profile(state: dict) -> dict:
    """Non-outcome description of a decision point, safe to show as 'similar patient'."""
    return {
        "age": round(state["age_years"]) if state.get("age_years") is not None else None,
        "stage": state.get("stage"), "grade": state.get("grade"), "residual": state.get("residual"),
        "line_number": state.get("line_number"), "setting": state.get("setting"),
        "months_since_last_platinum": (round(state["months_since_last_platinum"], 1)
                                       if state.get("months_since_last_platinum") is not None else None),
        "prior_classes": sorted(state.get("prior_exposure") or {}),
    }


class TwinDatabase:
    def __init__(self, entries: list[dict], mean: np.ndarray, scale: np.ndarray):
        self.entries = entries
        self.mean, self.scale = mean, scale
        self.matrix = (np.array([e["vector"] for e in entries]) - mean) / scale

    @classmethod
    def build(cls, histories: list[dict], ensemble: OutcomeEnsemble) -> "TwinDatabase":
        if not ensemble.member_seeds:
            raise ValueError("The outcome ensemble has no member seeds; out-of-bag predictions cannot be formed")
        train = [p for p in histories if p["split"] == "train" and p["lines"]]
        patients = sorted(p["patient_id"] for p in train)
        in_bag = [set(bootstrap_patients(patients, seed)) for seed in ensemble.member_seeds]
        entries = []
        for patient in train:
            oob = [k for k, bag in enumerate(in_bag) if patient["patient_id"] not in bag]
            lines = patient["lines"]
            pred = ensemble.predict([l["state"] for l in lines], [l["regimen"] for l in lines], oob) if oob else None
            for i, line in enumerate(lines):
                o = line["outcome"]
                entries.append({
                    "patient_id": patient["patient_id"], "line_index": line["line_index"],
                    "vector": encode_state(line["state"]), "regimen": line["regimen"],
                    "profile": line_profile(line["state"]),
                    "observed": {"response": o["response"],
                                 "pfs_months": o["pfs_months"], "pfs_event": o["pfs_event"],
                                 "os_months": o["os_months"], "os_event": o["os_event"]},
                    "oob_members": len(oob),
                    "oob": None if pred is None else {
                        "response": float(pred["response"][:, i].mean()),
                        "pfs": pred["pfs"][:, i, :].mean(0).tolist(),
                        "os": pred["os"][:, i, :].mean(0).tolist(),
                    },
                })
        vectors = np.array([e["vector"] for e in entries])
        mean = vectors.mean(0)
        scale = vectors.std(0)
        scale[scale < 1e-9] = 1.0       # constant features contribute nothing to distance
        return cls(entries, mean, scale)

    def nearest(self, state: dict, k: int = K_NEIGHBORS, exclude: str | None = None) -> list[dict]:
        """k nearest stored decision points from distinct patients (root-mean-square z distance)."""
        z = (np.array(encode_state(state)) - self.mean) / self.scale
        distances = np.sqrt(((self.matrix - z) ** 2).mean(1))
        chosen, seen = [], set()
        for index in np.argsort(distances, kind="stable"):
            entry = self.entries[index]
            if entry["patient_id"] in seen or entry["patient_id"] == exclude or entry["oob"] is None:
                continue
            seen.add(entry["patient_id"])
            chosen.append({"entry": entry, "index": int(index), "distance": float(distances[index]),
                           "similarity": float(1.0 / (1.0 + distances[index]))})
            if len(chosen) == k:
                break
        return chosen

    def projection(self) -> tuple[np.ndarray, callable]:
        """2-D map of the stored twins (first two principal components, scaled to [0, 1]) and a projector."""
        centred = self.matrix - self.matrix.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axes = vt[:2].T
        points = centred @ axes
        low, span = points.min(0), np.ptp(points, 0) + 1e-9

        def project(state: dict) -> np.ndarray:
            z = (np.array(encode_state(state)) - self.mean) / self.scale
            return np.clip(((z - self.matrix.mean(0)) @ axes - low) / span, 0.0, 1.0)
        return (points - low) / span, project

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state_features": STATE_FEATURES, "mean": self.mean.tolist(),
                                    "scale": self.scale.tolist(), "entries": self.entries}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TwinDatabase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["state_features"] != STATE_FEATURES:
            raise ValueError("Twin database was built with a different state schema; rebuild it")
        return cls(data["entries"], np.array(data["mean"]), np.array(data["scale"]))


def expected_events(hazards: np.ndarray, months: np.ndarray, events: np.ndarray, shift: float = 0.0) -> float:
    """Model-expected events over the observed follow-up, counted exactly as the training likelihood does.

    A bin counts if it was fully survived, and the bin of an event counts too;
    the bin in which follow-up was censored does not.
    """
    h = _shift(np.asarray(hazards, dtype=float), shift)
    bins = np.array([time_bin(m) for m in months])
    index = np.arange(h.shape[-1])
    at_risk = (index[None, :] < bins[:, None]) | ((index[None, :] == bins[:, None]) & (np.asarray(events)[:, None] > 0))
    return float((h * at_risk).sum())


def observed_expected_shift(evidence: list[tuple], prior_events: float) -> float:
    """Logit hazard shift that makes expected events match the shrunken observed/expected ratio."""
    if not evidence or math.isinf(prior_events):
        return 0.0
    events = np.array([e for e, _, _ in evidence])
    months = np.array([t for _, t, _ in evidence])
    hazards = np.array([h for _, _, h in evidence])
    expected = expected_events(hazards, months, events)
    if expected <= 0:
        return 0.0
    target = expected * (events.sum() + prior_events) / (expected + prior_events)
    low, high = -4.0, 4.0
    for _ in range(40):
        mid = (low + high) / 2.0
        low, high = (mid, high) if expected_events(hazards, months, events, mid) < target else (low, mid)
    return (low + high) / 2.0


def pooled_offsets(evidence: dict[str, list], prior_events: dict[str, float]) -> dict[str, float]:
    """Hazard offsets for progression and survival; response stays at the shared model."""
    return {"response": 0.0,
            "pfs": observed_expected_shift(evidence["pfs"], prior_events["pfs"]),
            "os": observed_expected_shift(evidence["os"], prior_events["os"])}


def gather_evidence(neighbors: list[dict], own_lines: list[dict], own_predictions: dict | None,
                    today_day: int | None = None) -> dict[str, list]:
    """(events, months followed, hazards) from neighbours (out-of-bag) and the patient's own lines up to today."""
    evidence = {"pfs": [], "os": []}
    for n in neighbors:
        observed, oob = n["entry"]["observed"], n["entry"]["oob"]
        evidence["pfs"].append((observed["pfs_event"], observed["pfs_months"], oob["pfs"]))
        evidence["os"].append((observed["os_event"], observed["os_months"], oob["os"]))
    for i, line in enumerate(own_lines):
        o = line["outcome"]
        evidence["pfs"].append((o["pfs_event"], o["pfs_months"], own_predictions["pfs"][:, i, :].mean(0)))
        if today_day is not None:
            # The patient is alive today: survival from that line is known only up to today.
            months_alive = max(0.0, (today_day - line["start_day"]) / DAYS_PER_MONTH)
            evidence["os"].append((0, months_alive, own_predictions["os"][:, i, :].mean(0)))
    return evidence


def create_twin(patient: dict, ensemble: OutcomeEnsemble, toxicity: ToxicityModel, database: TwinDatabase,
                current_state: dict | None = None, start_line: int = 0, k: int = K_NEIGHBORS,
                prior_events: dict[str, float] | None = None, members: list[int] | None = None) -> dict:
    """Build a new twin for `patient` from its k nearest stored twins (and its own earlier lines)."""
    if current_state is not None:
        state, own_lines = current_state, patient["lines"]
        today_day = round(current_state["months_since_diagnosis"] * DAYS_PER_MONTH)
    else:
        state, own_lines = patient["lines"][start_line]["state"], patient["lines"][:start_line]
        today_day = patient["lines"][start_line]["start_day"]
    members = members if members is not None else list(range(len(ensemble)))
    neighbors = database.nearest(state, k, exclude=patient["patient_id"])
    own_pred = (ensemble.predict([l["state"] for l in own_lines], [l["regimen"] for l in own_lines], members)
                if own_lines else None)
    offsets = pooled_offsets(gather_evidence(neighbors, own_lines, own_pred, today_day), prior_events or PRIOR_EVENTS)
    twin = PatientDigitalTwin(patient, ensemble, toxicity, start_line, members,
                              current_state=current_state, offsets=offsets)
    return {"twin": twin, "neighbors": neighbors, "offsets": offsets}


def evaluate_neighbor_twins(histories: list[dict], ensemble: OutcomeEnsemble, database: TwinDatabase, split: str,
                            strengths=(1.0, 2.0, 5.0, 10.0, math.inf), k: int = K_NEIGHBORS) -> dict:
    """Held-out comparison of the shared model alone (inf) with neighbour-built twins at each prior strength.

    A strength is a number (same for progression and survival) or a {"pfs": ..., "os": ...} dict.
    Scored on 12-month progression and death (Brier, AUROC) at every observed decision point of the split.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    patients = [p for p in histories if p["split"] == split and p["lines"]]
    rows = []
    for patient in patients:
        lines = patient["lines"]
        pred = ensemble.predict([l["state"] for l in lines], [l["regimen"] for l in lines])
        for i, line in enumerate(lines):
            neighbors = database.nearest(line["state"], k, exclude=patient["patient_id"])
            evidence = gather_evidence(neighbors, lines[:i], pred if i else None, line["start_day"])
            rows.append({"line": line, "i": i, "pred": pred, "evidence": evidence})
    results = {}
    for strength in strengths:
        y = {"pfs12": [], "os12": []}
        p = {"pfs12": [], "os12": []}
        priors = strength if isinstance(strength, dict) else {"pfs": strength, "os": strength}
        for row in rows:
            offsets = pooled_offsets(row["evidence"], priors)
            o, i, pred = row["line"]["outcome"], row["i"], row["pred"]
            for key, label, hazards in (("pfs12", twelve_month_label(o["pfs_months"], o["pfs_event"]), pred["pfs"]),
                                        ("os12", twelve_month_label(o["os_months"], o["os_event"]), pred["os"])):
                if label is not None:
                    shifted = _shift(hazards[:, i, :], offsets["pfs" if key == "pfs12" else "os"])
                    y[key].append(label)
                    p[key].append(float(1.0 - survival_curve(shifted)[:, TWELVE].mean()))
        metrics = {}
        for key in y:
            ys, ps = np.array(y[key]), np.array(p[key])
            metrics[key] = {"n": int(len(ys)), "brier": float(brier_score_loss(ys, ps)),
                            "auroc": float(roc_auc_score(ys, ps)) if 0 < ys.sum() < len(ys) else None,
                            "mean_predicted": float(ps.mean()), "observed_rate": float(ys.mean())}
        if isinstance(strength, dict):
            label = "chosen: " + ", ".join(f"{key} prior_events={value:g}" for key, value in strength.items())
        else:
            label = "shared model only (no personalisation)" if math.isinf(strength) else f"prior_events={strength:g}"
        results[label] = metrics
    return {"split": split, "k": k, "decision_points": len(rows), "results": results}
