"""Learned outcome model: what happens after a treatment line, given state and regimen.

An ensemble of small multi-task networks is trained on observed treatment
lines. Each member predicts, from the pre-treatment state and the regimen:

* tumor response (probability the next assessment is tumor-free);
* progression-free survival (first of progression or death), as discrete-time
  hazards over TIME_BINS, plus, per bin, the probability that an event in that
  bin is death (early first events are mostly progression, late ones often
  unrecorded progression followed by death);
* overall survival, as discrete-time hazards over TIME_BINS.

Censored follow-up is handled by the discrete-time survival likelihood. Each
member sees a different bootstrap of training patients, so member disagreement
is an uncertainty signal. Associations are observational: confounding by
indication is not removed.
"""
from __future__ import annotations

import bisect
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.paper_method_pipeline import ACTION_CLASSES

TIME_BINS = [0.0, 3.0, 6.0, 12.0, 24.0, 48.0]   # months; last bin is open-ended
HORIZON_MONTHS = 60.0
STATE_FEATURES = [
    "age_norm", "stage_norm", "stage_missing", "grade_norm", "grade_missing",
    "residual_norm", "residual_missing", "line_norm", "months_since_diagnosis_norm",
    "setting_primary", "setting_recurrent", "setting_progressive", "setting_unknown",
    "tumor_free", "tumor_with", "tumor_missing",
    "ecog_norm", "ecog_missing", "karnofsky_norm", "karnofsky_missing",
    *[f"prior_{c}" for c in ACTION_CLASSES],
    "platinum_free_norm", "platinum_resistant", "no_prior_platinum",
    "treatment_free_norm", "first_line",
]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def encode_state(state: dict) -> list[float]:
    age = state.get("age_years")
    stage, grade, residual = state.get("stage"), state.get("grade"), state.get("residual")
    setting = state.get("setting") or "unknown"
    tumor = state.get("last_tumor_status")
    ecog, kps = state.get("ecog"), state.get("karnofsky")
    exposure = state.get("prior_exposure") or {}
    pfi = state.get("months_since_last_platinum")
    tfi = state.get("months_since_last_line")
    vector = [
        _clip01((age - 30.0) / 60.0) if age is not None else 0.5,
        stage / 4.0 if stage is not None else 0.0, float(stage is None),
        grade / 3.0 if grade is not None else 0.0, float(grade is None),
        residual / 3.0 if residual is not None else 0.0, float(residual is None),
        min(state.get("line_number", 1), 7) / 7.0,
        _clip01(state.get("months_since_diagnosis", 0.0) / 120.0),
        float(setting == "primary"), float(setting == "recurrent"),
        float(setting == "progressive"), float(setting not in ("primary", "recurrent", "progressive")),
        float(tumor == "TF"), float(tumor == "WT"), float(tumor is None),
        _clip01(ecog / 4.0) if ecog is not None else 0.0, float(ecog is None),
        _clip01(kps / 100.0) if kps is not None else 0.0, float(kps is None),
        *[float(exposure.get(c, 0) > 0) for c in ACTION_CLASSES],
        _clip01(pfi / 36.0) if pfi is not None else 0.0,
        float(pfi is not None and pfi < 6.0), float(pfi is None),
        _clip01(tfi / 36.0) if tfi is not None else 0.0, float(tfi is None),
    ]
    assert len(vector) == len(STATE_FEATURES)
    return vector


def encode_regimen(regimen: list[str]) -> list[float]:
    return [float(c in regimen) for c in ACTION_CLASSES]


def time_bin(months: float) -> int:
    return min(len(TIME_BINS) - 1, max(0, bisect.bisect_right(TIME_BINS, months) - 1))


def survival_curve(hazards: np.ndarray) -> np.ndarray:
    """Probability of remaining event-free at the end of each bin (last axis)."""
    return np.cumprod(1.0 - hazards, axis=-1)


def restricted_mean_months(hazards: np.ndarray, horizon: float = HORIZON_MONTHS) -> np.ndarray:
    """Expected event-free months within `horizon`, with survival falling linearly inside each bin.

    The open-ended last bin is closed at HORIZON_MONTHS; shorter horizons truncate bins.
    """
    starts = np.array(TIME_BINS)
    widths = np.diff(TIME_BINS + [HORIZON_MONTHS])
    exposure = np.clip(horizon - starts, 0.0, widths)
    before = np.concatenate([np.ones(hazards.shape[:-1] + (1,)), survival_curve(hazards)[..., :-1]], axis=-1)
    return np.sum(before * (exposure - hazards * exposure ** 2 / (2.0 * widths)), axis=-1)


class OutcomeNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        bins = len(TIME_BINS)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.response = nn.Linear(hidden, 1)
        self.pfs = nn.Linear(hidden, bins)
        self.death_cause = nn.Linear(hidden, bins)
        self.os = nn.Linear(hidden, bins)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(torch.cat([state, action], dim=-1))
        return {"response": self.response(h).squeeze(-1), "pfs": self.pfs(h),
                "death_cause": self.death_cause(h), "os": self.os(h)}


def discrete_survival_nll(logits: torch.Tensor, bins: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of event/censoring bins under per-bin hazards."""
    index = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
    survived = (index < bins.unsqueeze(1)).float()
    event_at = ((index == bins.unsqueeze(1)) & (events.unsqueeze(1) > 0.5)).float()
    log_lik = (survived * F.logsigmoid(-logits)).sum(-1) + (event_at * F.logsigmoid(logits)).sum(-1)
    return -log_lik.mean()


def line_rows(histories: list[dict], splits: set[str]) -> list[dict]:
    rows = []
    for patient in histories:
        if patient["split"] not in splits:
            continue
        for line in patient["lines"]:
            rows.append({"patient_id": patient["patient_id"], "state": line["state"],
                         "regimen": line["regimen"], "outcome": line["outcome"]})
    return rows


def tensorize(rows: list[dict]) -> dict[str, torch.Tensor]:
    outcome = [r["outcome"] for r in rows]
    return {
        "state": torch.tensor([encode_state(r["state"]) for r in rows], dtype=torch.float32),
        "action": torch.tensor([encode_regimen(r["regimen"]) for r in rows], dtype=torch.float32),
        "response": torch.tensor([o["response"] if o["response"] is not None else -1 for o in outcome], dtype=torch.float32),
        "pfs_bin": torch.tensor([time_bin(o["pfs_months"]) for o in outcome], dtype=torch.long),
        "pfs_event": torch.tensor([o["pfs_event"] for o in outcome], dtype=torch.float32),
        "pfs_death": torch.tensor([o["pfs_event_is_death"] if o["pfs_event"] else -1 for o in outcome],
                                  dtype=torch.float32),
        "os_bin": torch.tensor([time_bin(o["os_months"]) for o in outcome], dtype=torch.long),
        "os_event": torch.tensor([o["os_event"] for o in outcome], dtype=torch.float32),
    }


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labelled = labels >= 0
    return (F.binary_cross_entropy_with_logits(logits[labelled], labels[labelled])
            if labelled.any() else logits.sum() * 0.0)


def multitask_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    death_cause_at_event = out["death_cause"].gather(1, batch["pfs_bin"].unsqueeze(1)).squeeze(1)
    return (_masked_bce(out["response"], batch["response"])
            + discrete_survival_nll(out["pfs"], batch["pfs_bin"], batch["pfs_event"])
            + _masked_bce(death_cause_at_event, batch["pfs_death"])
            + discrete_survival_nll(out["os"], batch["os_bin"], batch["os_event"]))


def _subset(data: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[index] for k, v in data.items()}


def train_member(train_rows: list[dict], val: dict[str, torch.Tensor], seed: int,
                 hidden: int, epochs: int, patience: int) -> tuple[OutcomeNet, dict]:
    if not train_rows or val["state"].shape[0] == 0:
        raise ValueError("Outcome-model training needs non-empty train and validation splits")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    # Bootstrap by patient so all of one patient's lines stay together.
    patients = sorted({r["patient_id"] for r in train_rows})
    by_patient: dict[str, list[dict]] = {}
    for row in train_rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    sample = [row for _ in patients for row in by_patient[rng.choice(patients)]]
    train = tensorize(sample)
    model = OutcomeNet(len(STATE_FEATURES), len(ACTION_CLASSES), hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_loss, best_state, stale, history = math.inf, None, 0, []
    n = train["state"].shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, 128):
            batch = _subset(train, order[start:start + 128])
            loss = multitask_loss(model(batch["state"], batch["action"]), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(multitask_loss(model(val["state"], val["action"]), val))
        history.append(val_loss)
        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"Outcome-model member {seed} never produced a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "epochs_run": len(history), "best_validation_loss": best_loss}


def months_bucket(months: float) -> str:
    return "lt6" if months < 6.0 else "6to12" if months < 12.0 else "ge12"


def transition_stats(histories: list[dict]) -> dict:
    """Empirical bookkeeping conventions of the training lines, used by the twin between lines.

    The outcome model predicts what happens during a line; how the *next*
    line's recorded state looks (disease setting, last tumor status, whether
    an end date exists, treatment duration) follows these frequencies so the
    simulated states are encoded the way the data records them.
    """
    from collections import Counter
    setting = {b: Counter() for b in ("lt6", "6to12", "ge12")}
    status = {"1": Counter(), "0": Counter()}
    durations, total, ends_known = [], 0, 0
    for patient in histories:
        if patient["split"] != "train":
            continue
        lines = patient["lines"]
        for line in lines:
            total += 1
            if line["end_day"] is not None:
                ends_known += 1
                durations.append((line["end_day"] - line["start_day"]) / 30.44)
        for prev, nxt in zip(lines, lines[1:]):
            setting[months_bucket(prev["outcome"]["pfs_months"])][nxt["state"]["setting"]] += 1
            if prev["outcome"]["response"] is not None:
                status[str(prev["outcome"]["response"])][str(nxt["state"]["last_tumor_status"])] += 1

    def normalise(counter):
        n = sum(counter.values())
        return {k: v / n for k, v in sorted(counter.items())} if n else {}
    return {
        "next_setting_by_months_to_event": {b: normalise(c) for b, c in setting.items()},
        "next_tumor_status_by_response": {r: normalise(c) for r, c in status.items()},
        "end_day_known_rate": ends_known / total if total else 0.0,
        "treatment_months_quantiles": [float(q) for q in np.quantile(durations, np.linspace(0, 1, 21))] if durations else [4.0],
    }


class OutcomeEnsemble:
    def __init__(self, models: list[OutcomeNet], hidden: int, transitions: dict | None = None):
        self.models = models
        self.hidden = hidden
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.models)

    @torch.no_grad()
    def predict(self, states: list[dict], regimens: list[list[str]], members: list[int] | None = None) -> dict[str, np.ndarray]:
        """Per-member probabilities: response [M, N]; pfs/os hazards and death_cause [M, N, bins]."""
        s = torch.tensor([encode_state(x) for x in states], dtype=torch.float32)
        a = torch.tensor([encode_regimen(x) for x in regimens], dtype=torch.float32)
        chosen = members if members is not None else range(len(self.models))
        outs = [self.models[i](s, a) for i in chosen]
        return {key: torch.stack([torch.sigmoid(o[key]) for o in outs]).numpy()
                for key in ("response", "pfs", "death_cause", "os")}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden": self.hidden, "state_features": STATE_FEATURES, "action_classes": ACTION_CLASSES,
                    "time_bins": TIME_BINS, "transitions": self.transitions,
                    "model_state_dicts": [m.state_dict() for m in self.models]}, path)

    @classmethod
    def load(cls, path: Path) -> "OutcomeEnsemble":
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["state_features"] != STATE_FEATURES or checkpoint["action_classes"] != ACTION_CLASSES:
            raise ValueError("Outcome-model checkpoint schema does not match the current code")
        if checkpoint["time_bins"] != TIME_BINS:
            raise ValueError("Outcome-model checkpoint time bins do not match the current code")
        models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = OutcomeNet(len(STATE_FEATURES), len(ACTION_CLASSES), checkpoint["hidden"])
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        return cls(models, checkpoint["hidden"], checkpoint.get("transitions"))


def train_ensemble(histories: list[dict], members: int = 10, hidden: int = 64, epochs: int = 400,
                   patience: int = 40, seed: int = 42) -> tuple[OutcomeEnsemble, list[dict]]:
    train_rows = line_rows(histories, {"train"})
    val = tensorize(line_rows(histories, {"validation"}))
    models, reports = [], []
    for k in range(members):
        model, report = train_member(train_rows, val, seed + k, hidden, epochs, patience)
        models.append(model)
        reports.append(report)
    return OutcomeEnsemble(models, hidden, transition_stats(histories)), reports


def concordance_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float | None:
    """Harrell's C: higher risk should mean an earlier observed event."""
    concordant = comparable = 0.0
    for i in range(len(times)):
        if not events[i]:
            continue
        later = times > times[i]
        comparable += later.sum()
        concordant += (risks[i] > risks[later]).sum() + 0.5 * (risks[i] == risks[later]).sum()
    return float(concordant / comparable) if comparable else None


def evaluate_ensemble(ensemble: OutcomeEnsemble, histories: list[dict], split: str = "test",
                      members: list[int] | None = None, seed: int = 42) -> dict:
    """Held-out discrimination/calibration plus a check that the regimen carries signal."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    rows = line_rows(histories, {split})
    train_rows = line_rows(histories, {"train"})
    pred = ensemble.predict([r["state"] for r in rows], [r["regimen"] for r in rows], members)
    response_p = pred["response"].mean(0)
    pfs_s12 = survival_curve(pred["pfs"].mean(0))[:, time_bin(12.0) - 1]
    os_s12 = survival_curve(pred["os"].mean(0))[:, time_bin(12.0) - 1]
    labelled = np.array([r["outcome"]["response"] is not None for r in rows])
    y = np.array([r["outcome"]["response"] or 0 for r in rows])[labelled]

    # Baselines for response: training prevalence and a state+regimen logistic model.
    train_lab = [r for r in train_rows if r["outcome"]["response"] is not None]
    prevalence = float(np.mean([r["outcome"]["response"] for r in train_lab]))
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(
        [encode_state(r["state"]) + encode_regimen(r["regimen"]) for r in train_lab],
        [r["outcome"]["response"] for r in train_lab])
    lr_p = lr.predict_proba([encode_state(r["state"]) + encode_regimen(r["regimen"])
                             for r, keep in zip(rows, labelled) if keep])[:, 1]

    def survival_metrics(kind: str, s12: np.ndarray) -> dict:
        times = np.array([r["outcome"][f"{kind}_months"] for r in rows])
        events = np.array([r["outcome"][f"{kind}_event"] for r in rows])
        known = (times >= 12.0) | (events == 1)
        observed_12 = ((times < 12.0) & (events == 1))[known]
        return {
            "lines": int(len(rows)), "events": int(events.sum()),
            "c_index_12m_risk": concordance_index(times, events, 1.0 - s12),
            "predicted_event_rate_12m": float(np.mean(1.0 - s12[known])) if known.any() else None,
            "observed_event_rate_12m_known_status": float(observed_12.mean()) if known.any() else None,
        }

    # Regimen signal: does the held-out likelihood get worse when regimens are shuffled?
    def nll(regimens: list[list[str]]) -> float:
        batch = tensorize([dict(r, regimen=g) for r, g in zip(rows, regimens)])
        chosen = members if members is not None else range(len(ensemble))
        with torch.no_grad():
            return float(np.mean([float(multitask_loss(ensemble.models[i](batch["state"], batch["action"]), batch))
                                  for i in chosen]))
    rng = random.Random(seed)
    true_nll = nll([r["regimen"] for r in rows])
    shuffled = []
    for _ in range(20):
        regimens = [r["regimen"] for r in rows]
        rng.shuffle(regimens)
        shuffled.append(nll(regimens))

    response = {"labelled_lines": int(labelled.sum()), "tumor_free": int(y.sum())}
    if labelled.sum() and 0 < y.sum() < labelled.sum():
        response.update({
            "auroc_twin": float(roc_auc_score(y, response_p[labelled])),
            "auroc_logistic_baseline": float(roc_auc_score(y, lr_p)),
            "brier_twin": float(brier_score_loss(y, response_p[labelled])),
            "brier_prevalence_baseline": float(brier_score_loss(y, np.full(len(y), prevalence))),
            "brier_logistic_baseline": float(brier_score_loss(y, lr_p)),
            "mean_predicted": float(response_p[labelled].mean()), "observed_rate": float(y.mean()),
        })
    event_rows = [i for i, r in enumerate(rows) if r["outcome"]["pfs_event"]]
    cause_y = np.array([rows[i]["outcome"]["pfs_event_is_death"] for i in event_rows])
    cause_p = np.array([pred["death_cause"].mean(0)[i, time_bin(rows[i]["outcome"]["pfs_months"])]
                        for i in event_rows])
    death_cause = {"pfs_events": len(event_rows), "deaths_first": int(cause_y.sum()),
                   "mean_predicted": float(cause_p.mean()) if len(cause_p) else None,
                   "observed_rate": float(cause_y.mean()) if len(cause_y) else None}
    if 0 < cause_y.sum() < len(cause_y):
        death_cause["auroc"] = float(roc_auc_score(cause_y, cause_p))
    return {
        "split": split, "lines": len(rows),
        "response": response,
        "death_as_first_event": death_cause,
        "progression_free_survival": survival_metrics("pfs", pfs_s12),
        "overall_survival": survival_metrics("os", os_s12),
        "regimen_signal": {
            "heldout_nll_true_regimens": true_nll,
            "heldout_nll_shuffled_regimens_mean": float(np.mean(shuffled)),
            "nll_increase_when_shuffled": float(np.mean(shuffled) - true_nll),
            "interpretation": "Positive means the observed regimen carries predictive signal beyond the state; it is not a causal effect estimate.",
        },
        "mean_member_std_response": float(pred["response"].std(0).mean()),
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
