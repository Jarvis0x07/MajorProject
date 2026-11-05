"""Step 2: the outcome model, which predicts what happens after a treatment line.

Given the state before a line and the regimen, it predicts:

    response   chance the tumour disappears (next check-up tumour-free)
    pfs        progression-free survival: risk of progression or death in each time bin
    death      for an event in each bin, the chance it is death rather than progression
    os         overall survival: risk of death in each time bin

Survival is modelled as a risk per time bin (0-3, 3-6, 6-12, 12-24, 24-48, 48+ months), so
patients still alive at last contact (censored) are used correctly. It is an ensemble of 10
small networks, each trained on a different bootstrap sample of the training patients; how much
they disagree measures how unsure the prediction is.
"""
from __future__ import annotations

import bisect
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import DRUG_CLASSES, SETTINGS

# ---------------------------------------------------------------------------------------------
# Settings: time bins for survival, the 5-year horizon, ensemble size, and the input features.
# ---------------------------------------------------------------------------------------------
TIME_BINS = [0.0, 3.0, 6.0, 12.0, 24.0, 48.0]   # months; the last bin is open-ended
HORIZON = 60.0                                  # every expectation is "within 5 years"
MEMBERS = 10

# Names of the 29 numbers that describe a patient's state (same order as encode_state).
FEATURES = ["age", "stage", "stage_missing", "grade", "grade_missing", "residual", "residual_missing",
            "line_number", "months_since_diagnosis", *[f"setting_{s}" for s in SETTINGS],
            *[f"had_{c}" for c in DRUG_CLASSES],
            "months_since_platinum", "platinum_resistant", "no_platinum_yet", "months_since_last_line", "first_line"]


# ---------------------------------------------------------------------------------------------
# Encoding: turn a patient state and a regimen into numbers the network can read.
# ---------------------------------------------------------------------------------------------
def _clip(value: float) -> float:
    # Keep a number between 0 and 1.
    return min(1.0, max(0.0, value))


def encode_state(state: dict) -> list[float]:
    """The state as numbers in [0, 1]; missing values get their own indicator."""
    age, stage, grade, residual = (state.get(k) for k in ("age_years", "stage", "grade", "residual"))
    platinum, last_line = state.get("months_since_last_platinum"), state.get("months_since_last_line")
    prior = state.get("prior_exposure") or {}
    setting = state.get("setting") or "unknown"
    return [
        # Age scaled so 30 years -> 0 and 90 years -> 1 (unknown age -> middle value).
        _clip((age - 30.0) / 60.0) if age is not None else 0.5,
        # Stage, grade and residual tumour, each with a "was it missing?" flag.
        stage / 4.0 if stage is not None else 0.0, float(stage is None),
        grade / 3.0 if grade is not None else 0.0, float(grade is None),
        residual / 3.0 if residual is not None else 0.0, float(residual is None),
        # Which treatment line this is, and how long since diagnosis (10 years -> 1).
        min(state["line_number"], 7) / 7.0,
        _clip(state["months_since_diagnosis"] / 120.0),
        # Disease setting as four yes/no values (newly diagnosed, recurrent, progressive, unknown).
        *[float(setting == s) for s in SETTINGS],
        # One yes/no value per drug class: was it given in an earlier line?
        *[float(prior.get(c, 0) > 0) for c in DRUG_CLASSES],
        # Time since the last platinum (3 years -> 1), platinum-resistant (< 6 months), never had platinum.
        _clip(platinum / 36.0) if platinum is not None else 0.0,
        float(platinum is not None and platinum < 6.0), float(platinum is None),
        # Time since the previous line ended, and "this is the first line".
        _clip(last_line / 36.0) if last_line is not None else 0.0, float(last_line is None),
    ]


def encode_regimen(regimen: list[str]) -> list[float]:
    # One yes/no value per drug class in the regimen.
    return [float(c in regimen) for c in DRUG_CLASSES]


# ---------------------------------------------------------------------------------------------
# Survival helpers: which bin a time falls in, survival curves and expected months.
# ---------------------------------------------------------------------------------------------
def time_bin(months: float) -> int:
    # Index of the time bin that contains `months` (e.g. 8 months -> bin 2, the 6-12 bin).
    return min(len(TIME_BINS) - 1, max(0, bisect.bisect_right(TIME_BINS, months) - 1))


def survival(hazards: np.ndarray) -> np.ndarray:
    """Chance of still being event-free at the end of each bin."""
    # Surviving to the end of a bin = surviving every bin so far: multiply (1 - risk) bin by bin.
    return np.cumprod(1.0 - hazards, axis=-1)


def expected_months(hazards: np.ndarray, horizon: float = HORIZON) -> np.ndarray:
    """Expected event-free months within `horizon` (survival falls linearly inside each bin)."""
    # This is the area under the survival curve up to the horizon. For each bin: the chance of
    # being event-free when it starts, times the part of the bin that lies before the horizon,
    # minus the time lost by events during the bin.
    starts = np.array(TIME_BINS)
    widths = np.diff(TIME_BINS + [HORIZON])
    exposure = np.clip(horizon - starts, 0.0, widths)
    before = np.concatenate([np.ones(hazards.shape[:-1] + (1,)), survival(hazards)[..., :-1]], axis=-1)
    return np.sum(before * (exposure - hazards * exposure ** 2 / (2.0 * widths)), axis=-1)


# ---------------------------------------------------------------------------------------------
# The neural network: a shared body with four output heads, one per prediction.
# ---------------------------------------------------------------------------------------------
class OutcomeNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        bins = len(TIME_BINS)
        # Shared body: state + regimen -> 64 hidden numbers (dropout reduces overfitting).
        self.trunk = nn.Sequential(nn.Linear(len(FEATURES) + len(DRUG_CLASSES), hidden), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2))
        # Heads: response (1 number), and one risk per time bin for progression, death-as-cause and survival.
        self.heads = nn.ModuleDict({"response": nn.Linear(hidden, 1), "pfs": nn.Linear(hidden, bins),
                                    "death": nn.Linear(hidden, bins), "os": nn.Linear(hidden, bins)})

    def forward(self, state: torch.Tensor, regimen: torch.Tensor) -> dict[str, torch.Tensor]:
        # Run the shared body once, then every head. Outputs are raw scores (sigmoid gives probabilities).
        h = self.trunk(torch.cat([state, regimen], dim=-1))
        out = {name: head(h) for name, head in self.heads.items()}
        out["response"] = out["response"].squeeze(-1)
        return out


# ---------------------------------------------------------------------------------------------
# Training data and loss: what the network is asked to get right.
# ---------------------------------------------------------------------------------------------
def _tensors(lines: list[dict]) -> dict[str, torch.Tensor]:
    # Turn a list of treatment lines into tensors. Unknown labels are stored as -1 and skipped in the loss.
    o = [line["outcome"] for line in lines]
    return {
        "state": torch.tensor([encode_state(l["state"]) for l in lines], dtype=torch.float32),
        "regimen": torch.tensor([encode_regimen(l["regimen"]) for l in lines], dtype=torch.float32),
        "response": torch.tensor([-1 if x["response"] is None else x["response"] for x in o], dtype=torch.float32),
        "pfs_bin": torch.tensor([time_bin(x["pfs_months"]) for x in o]),
        "pfs_event": torch.tensor([x["pfs_event"] for x in o], dtype=torch.float32),
        "death": torch.tensor([x["pfs_event_is_death"] if x["pfs_event"] else -1 for x in o], dtype=torch.float32),
        "os_bin": torch.tensor([time_bin(x["os_months"]) for x in o]),
        "os_event": torch.tensor([x["os_event"] for x in o], dtype=torch.float32),
    }


def _survival_nll(logits: torch.Tensor, bins: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Every bin before the last one followed was survived; the last one holds the event, if there was one."""
    # Example: died at 8 months -> survived bins 0-3 and 3-6, event in bin 6-12.
    #          alive at 8 months (censored) -> survived bins 0-3 and 3-6; nothing is said about 6-12.
    # The loss is minus the log-probability of exactly that happening (smaller = better fit).
    index = torch.arange(logits.shape[-1]).unsqueeze(0)
    survived = (index < bins.unsqueeze(1)).float()
    event = ((index == bins.unsqueeze(1)) & (events.unsqueeze(1) > 0.5)).float()
    return -((survived * F.logsigmoid(-logits)).sum(-1) + (event * F.logsigmoid(logits)).sum(-1)).mean()


def _labelled_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Yes/no loss (binary cross-entropy) on the rows whose label is known.
    keep = labels >= 0
    return F.binary_cross_entropy_with_logits(logits[keep], labels[keep]) if keep.any() else logits.sum() * 0.0


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # "Was the first event death?" is scored in the bin where the event actually happened.
    death_at_event = out["death"].gather(1, batch["pfs_bin"].unsqueeze(1)).squeeze(1)
    # Total loss = response + progression-free survival + death-as-first-event + overall survival.
    return (_labelled_bce(out["response"], batch["response"])
            + _survival_nll(out["pfs"], batch["pfs_bin"], batch["pfs_event"])
            + _labelled_bce(death_at_event, batch["death"])
            + _survival_nll(out["os"], batch["os_bin"], batch["os_event"]))


# ---------------------------------------------------------------------------------------------
# Training one network on a bootstrap sample of the training patients.
# ---------------------------------------------------------------------------------------------
def training_patients(histories: list[dict]) -> list[str]:
    # Ids of training patients that have at least one treatment line (sorted, so always the same order).
    return sorted(p["patient_id"] for p in histories if p["split"] == "train" and p["lines"])


def bootstrap(patients: list[str], seed: int) -> list[str]:
    """The patients (drawn with replacement) that the network with this seed trains on."""
    # Some patients are drawn twice, others not at all ("out-of-bag"). The twins use the
    # out-of-bag networks to see how the model does on patients it never trained on.
    rng = random.Random(seed)
    return [rng.choice(patients) for _ in patients]


def _train_member(histories: list[dict], seed: int, epochs: int = 400, patience: int = 40) -> OutcomeNet:
    torch.manual_seed(seed)
    # Training data: all lines of the bootstrap-sampled patients. Validation data: validation patients.
    by_patient = {p["patient_id"]: p["lines"] for p in histories if p["split"] == "train"}
    train = _tensors([line for pid in bootstrap(training_patients(histories), seed) for line in by_patient[pid]])
    val = _tensors([line for p in histories if p["split"] == "validation" for line in p["lines"]])
    net = OutcomeNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_weights, stale = math.inf, None, 0
    n = len(train["state"])
    for _ in range(epochs):
        # One pass over the training data in shuffled batches of 128 lines.
        net.train()
        order = torch.randperm(n)
        for start in range(0, n, 128):
            batch = {k: v[order[start:start + 128]] for k, v in train.items()}
            loss = loss_fn(net(batch["state"], batch["regimen"]), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Check the validation loss; remember the best weights so far.
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(val["state"], val["regimen"]), val))
        if val_loss < best - 1e-4:
            best, best_weights, stale = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            # Early stopping: quit after 40 passes without improvement (the network starts to overfit).
            stale += 1
            if stale >= patience:
                break
    # Use the weights that did best on the validation patients.
    net.load_state_dict(best_weights)
    net.eval()
    return net


# ---------------------------------------------------------------------------------------------
# How the next line's state looks in the data, so the twin's simulated lines look like real ones.
# ---------------------------------------------------------------------------------------------
def transition_stats(histories: list[dict]) -> dict:
    """How the dataset records the next line's state, so simulated lines look like real ones.

    next_setting: disease setting of the next line, by how long the previous line lasted.
    end_date_known / treatment_months: whether a line has an end date, and how long treatment lasts.
    """
    settings = {bucket: Counter() for bucket in ("<6", "6-12", ">=12")}
    durations, total = [], 0
    for patient in histories:
        if patient["split"] != "train":
            continue
        lines = patient["lines"]
        # Count the next line's setting (recurrent / progressive / unknown) after short, medium and long lines.
        for prev, nxt in zip(lines, lines[1:]):
            settings[duration_bucket(prev["outcome"]["pfs_months"])][nxt["state"]["setting"]] += 1
        # Collect treatment durations of lines that have an end date.
        for line in lines:
            total += 1
            if line["end_day"] is not None:
                durations.append((line["end_day"] - line["start_day"]) / 30.44)
    # Turn counts into probabilities; summarise durations as 21 evenly spaced quantiles.
    return {"next_setting": {b: {s: n / sum(c.values()) for s, n in c.items()} for b, c in settings.items()},
            "end_date_known": len(durations) / total,
            "treatment_months": [float(q) for q in np.quantile(durations, np.linspace(0, 1, 21))]}


def duration_bucket(months: float) -> str:
    return "<6" if months < 6.0 else "6-12" if months < 12.0 else ">=12"


# ---------------------------------------------------------------------------------------------
# The ensemble: 10 trained networks used together.
# ---------------------------------------------------------------------------------------------
class OutcomeModel:
    def __init__(self, nets: list[OutcomeNet], seeds: list[int], transitions: dict):
        self.nets, self.seeds, self.transitions = nets, seeds, transitions

    def __len__(self) -> int:
        return len(self.nets)

    @torch.no_grad()
    def predict(self, states: list[dict], regimens: list[list[str]], members: list[int] | None = None) -> dict:
        """Per-network probabilities: response [M, N]; pfs, death and os risks [M, N, bins]."""
        # Encode the inputs, run each chosen network (all 10 by default), and turn scores into
        # probabilities. Results keep one row per network so callers can average or measure spread.
        s = torch.tensor([encode_state(x) for x in states], dtype=torch.float32)
        r = torch.tensor([encode_regimen(x) for x in regimens], dtype=torch.float32)
        outs = [self.nets[m](s, r) for m in (members if members is not None else range(len(self.nets)))]
        return {k: torch.sigmoid(torch.stack([o[k] for o in outs])).numpy() for k in ("response", "pfs", "death", "os")}

    def save(self, path: Path) -> None:
        # Save the feature list too, so a model trained with different features is never loaded by mistake.
        torch.save({"features": FEATURES, "seeds": self.seeds, "transitions": self.transitions,
                    "weights": [net.state_dict() for net in self.nets]}, path)

    @classmethod
    def load(cls, path: Path) -> "OutcomeModel":
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["features"] != FEATURES:
            raise ValueError("The saved outcome model uses different features; run `python -m digital_twin_rl train-model`")
        # Rebuild each network and put the saved weights back in.
        nets = []
        for weights in checkpoint["weights"]:
            net = OutcomeNet()
            net.load_state_dict(weights)
            nets.append(net.eval())
        return cls(nets, checkpoint["seeds"], checkpoint["transitions"])


def train_model(histories: list[dict], seed: int = 42) -> OutcomeModel:
    # Train 10 networks, each with its own seed (so its own bootstrap sample), then bundle them.
    seeds = [seed + k for k in range(MEMBERS)]
    nets = []
    for k, member_seed in enumerate(seeds):
        print(f"  training network {k + 1}/{MEMBERS}")
        nets.append(_train_member(histories, member_seed))
    return OutcomeModel(nets, seeds, transition_stats(histories))


# ---------------------------------------------------------------------------------------------
# Checking accuracy on patients the model never saw.
# ---------------------------------------------------------------------------------------------
def concordance(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    """Harrell's C-index: how often the patient with the earlier death was given the higher risk."""
    # Compare every patient who died with every patient followed for longer: a pair "agrees" when
    # the one who died first had the higher predicted risk. 0.5 = guessing, 1 = perfect ordering.
    agree = pairs = 0.0
    for i in np.flatnonzero(events):
        later = times > times[i]
        pairs += later.sum()
        agree += (risks[i] > risks[later]).sum() + 0.5 * (risks[i] == risks[later]).sum()
    return float(agree / pairs)


def evaluate_model(model: OutcomeModel, histories: list[dict], split: str = "test") -> dict:
    """Accuracy on patients the model never trained on."""
    from sklearn.metrics import roc_auc_score

    # Predict every treatment line of the held-out patients with the regimen they really got.
    lines = [line for p in histories if p["split"] == split for line in p["lines"]]
    pred = model.predict([l["state"] for l in lines], [l["regimen"] for l in lines])
    labelled = [i for i, l in enumerate(lines) if l["outcome"]["response"] is not None]
    times = np.array([l["outcome"]["os_months"] for l in lines])
    events = np.array([l["outcome"]["os_event"] for l in lines])
    # Predicted chance of death within 12 months (averaged over the 10 networks).
    death_12m = 1.0 - survival(pred["os"].mean(0))[:, time_bin(12.0) - 1]
    # Only patients whose 12-month status is known (died, or followed for at least a year) are compared.
    status_known = (times >= 12.0) | (events == 1)
    return {
        "split": split, "treatment_lines": len(lines),
        "response_auroc": float(roc_auc_score([lines[i]["outcome"]["response"] for i in labelled],
                                              pred["response"].mean(0)[labelled])),
        "survival_c_index": concordance(times, events, death_12m),
        "death_within_12m_predicted": float(death_12m[status_known].mean()),
        "death_within_12m_observed": float(((times < 12.0) & (events == 1))[status_known].mean()),
    }
