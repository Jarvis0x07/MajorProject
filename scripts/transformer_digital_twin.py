#!/usr/bin/env python3
"""Train/evaluate a paper-inspired Transformer ensemble on observed TCGA-OV sequences.

This implements the paper's Stage 1 dynamics model shape, not the complete
paper system. It predicts the next *recorded treatment-decision state* from a
patient's prior decision states and aligned 11-class treatment actions. Because
the local export has incomplete treatment histories, this is observational
forecasting and must not be interpreted as a causal counterfactual simulator.

Requires PyTorch. Example:
  python scripts/transformer_digital_twin.py \
      --input data/processed/paper_method/state_action_outcome_candidates.csv \
      --output-dir data/processed/transformer_twin
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # Make the optional dependency failure actionable.
    raise SystemExit(
        "PyTorch is required for the Transformer twin. Install it in this Python "
        "environment, then rerun this script."
    ) from exc


ACTION_CLASSES = (
    "platinum", "taxanes", "anthracyclines", "antimetabolites",
    "topoisomerase_inhibitors", "alkylating_agents", "anti_angiogenics",
    "hormonal_agents", "vinca_alkaloids", "radiation", "other",
)
FEATURE_NAMES = (
    "age_norm", "age_missing", "tumor_free", "tumor_missing",
    "grade_norm", "grade_missing", "stage_norm", "stage_missing",
    "cumulative_dated_drug_count_norm", "prior_radiation",
    "ecog_norm", "ecog_missing", "karnofsky_norm", "karnofsky_missing",
    "has_undated_treatment_records",
)
VALUE_MISSING_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13))


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalized(value: float | None, low: float, high: float) -> tuple[float, float]:
    if value is None:
        return 0.0, 1.0
    return min(1.0, max(0.0, (value - low) / (high - low))), 0.0


def grade_number(value: Any) -> float | None:
    text = str(value or "").strip().upper()
    match = re.search(r"(?:GRADE|G)\s*([1-4])\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:GRADE|G)\s*(IV|III|II|I)\b", text)
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return float(roman[match.group(1)]) if match else None


def stage_number(value: Any) -> float | None:
    text = str(value or "").strip().upper()
    match = re.search(r"(?:FIGO\s*)?STAGE\s*(IV|III|II|I)\b", text)
    if not match:
        match = re.search(r"\b(IV|III|II|I)[ABC]?\b", text)
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return float(roman[match.group(1)]) if match else None


def encode_state(row: dict[str, str]) -> list[float]:
    age, age_missing = normalized(number(row.get("state_age_years")), 18, 90)
    response = row.get("state_tumor_status_pre_action")
    response_missing = float(response not in ("TF-Tumor Free", "WT-With Tumor"))
    tumor_free = 1.0 if response == "TF-Tumor Free" else 0.0
    grade, grade_missing = normalized(grade_number(row.get("state_tumor_grade")), 1, 4)
    stage, stage_missing = normalized(stage_number(row.get("state_figo_stage")), 1, 4)
    drug_count, _ = normalized(number(row.get("state_cumulative_dated_drug_record_count")), 0, 20)
    radiation = float(number(row.get("state_radiation_history")) or 0 > 0)
    ecog, ecog_missing = normalized(number(row.get("state_ecog_pre_action")), 0, 5)
    kps, kps_missing = normalized(number(row.get("state_karnofsky_pre_action")), 0, 100)
    undated = float(number(row.get("state_has_undated_treatment_records")) or 0 > 0)
    values = [age, age_missing, tumor_free, response_missing, grade, grade_missing,
              stage, stage_missing, drug_count, radiation, ecog, ecog_missing,
              kps, kps_missing, undated]
    return [min(1.0, max(0.0, float(value))) for value in values]


def encode_action(row: dict[str, str]) -> list[float]:
    return [float(number(row.get(f"action_{name}")) or 0 > 0) for name in ACTION_CLASSES]


def make_sequences(rows: list[dict[str, str]], sequence_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if row.get("action_mapping_complete") == "1"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        day = number(row.get("decision_day"))
        patient = row.get("case_id")
        split = row.get("split")
        if day is not None and patient and split:
            grouped[(split, patient)].append(row)

    examples: list[dict[str, Any]] = []
    patient_counts: dict[str, set[str]] = defaultdict(set)
    split_counters: Counter[str] = Counter()
    incomplete_counters: Counter[str] = Counter()
    for (split, patient), patient_rows in grouped.items():
        patient_rows.sort(key=lambda item: number(item.get("decision_day")) or 0)
        states = [encode_state(row) for row in patient_rows]
        actions = [encode_action(row) for row in patient_rows]
        for next_index in range(1, len(patient_rows)):
            prev_day = number(patient_rows[next_index - 1].get("decision_day"))
            next_day = number(patient_rows[next_index].get("decision_day"))
            if prev_day is None or next_day is None or next_day <= prev_day:
                continue
            start = max(0, next_index - sequence_length)
            state_sequence = states[start:next_index]
            action_sequence = actions[start:next_index]
            padding = sequence_length - len(state_sequence)
            # Right-pad so every causal query, including padding positions,
            # can still attend to at least one earlier valid token. Left
            # padding plus a causal mask can create fully masked query rows
            # (NaNs in scaled dot-product attention on some PyTorch versions).
            padded_states = state_sequence + [[0.0] * len(FEATURE_NAMES) for _ in range(padding)]
            padded_actions = action_sequence + [[0.0] * len(ACTION_CLASSES) for _ in range(padding)]
            target = states[next_index]
            target_mask = [1.0] * len(FEATURE_NAMES)
            for value_index, missing_index in VALUE_MISSING_PAIRS:
                if target[missing_index] > 0.5:
                    target_mask[value_index] = 0.0
            examples.append({
                "split": split,
                "patient_key": patient,
                "input_states": padded_states,
                "input_actions": padded_actions,
                "padding_mask": [False] * len(state_sequence) + [True] * padding,
                "last_state": states[next_index - 1],
                "target_state": target,
                "target_mask": target_mask,
                "elapsed_days_to_next_decision": next_day - prev_day,
                "history_incomplete": int(patient_rows[next_index - 1].get("state_has_undated_treatment_records") == "1"),
            })
            split_counters[split] += 1
            patient_counts[split].add(patient)
            incomplete_counters[split] += examples[-1]["history_incomplete"]

    audit = {
        "eligible_decision_rows": len(eligible),
        "sequence_length": sequence_length,
        "transitions_by_split": dict(split_counters),
        "patients_with_transitions_by_split": {split: len(items) for split, items in patient_counts.items()},
        "transitions_with_undated_history_by_split": dict(incomplete_counters),
        "target_definition": "next strictly later recorded treatment-decision state, not an immediate post-treatment or causal outcome",
    }
    return examples, audit


class SequenceDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.examples[index]
        return {
            "states": torch.tensor(item["input_states"], dtype=torch.float32),
            "actions": torch.tensor(item["input_actions"], dtype=torch.float32),
            "padding_mask": torch.tensor(item["padding_mask"], dtype=torch.bool),
            "last_state": torch.tensor(item["last_state"], dtype=torch.float32),
            "target_state": torch.tensor(item["target_state"], dtype=torch.float32),
            "target_mask": torch.tensor(item["target_mask"], dtype=torch.float32),
        }


class TransformerDynamics(nn.Module):
    """Causal state/action Transformer with the paper's bounded residual head."""

    def __init__(self, state_dim: int, action_dim: int, sequence_length: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.sequence_length = sequence_length
        self.state_projection = nn.Linear(state_dim, d_model)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, state_dim))

    def forward(self, states: torch.Tensor, actions: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        length = states.shape[1]
        x = self.state_projection(states) + self.action_projection(actions) + self.position[:, :length]
        causal_mask = torch.triu(torch.ones(length, length, device=states.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        last_valid = (~padding_mask).sum(dim=1).clamp_min(1) - 1
        last_hidden = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_valid]
        last_state = states[torch.arange(states.shape[0], device=states.device), last_valid]
        # Paper Eq. (1): normalized bounded residual state update.
        delta = 0.05 * torch.tanh(self.head(last_hidden))
        return torch.clamp(last_state + delta, 0.0, 1.0)


def simulate_trajectory(models: list[TransformerDynamics], initial_states: list[list[float]],
                        prior_actions: list[list[float]], planned_actions: list[list[float]],
                        sequence_length: int, device: torch.device) -> dict[str, Any]:
    """Roll a state sequence forward under a caller-supplied action plan.

    `prior_actions` must contain the observed actions taken from all initial
    states except the current final state. Each action in `planned_actions` is
    applied to the current state. Outputs are model hypotheses and include
    ensemble disagreement; this helper does not claim counterfactual validity.
    """
    if not models or not initial_states or len(prior_actions) != len(initial_states) - 1:
        raise ValueError("Provide models, at least one initial state, and one prior action per earlier state")
    if any(len(state) != len(FEATURE_NAMES) for state in initial_states):
        raise ValueError("Initial state vectors do not match the declared feature schema")
    if any(len(action) != len(ACTION_CLASSES) for action in [*prior_actions, *planned_actions]):
        raise ValueError("Action vectors do not match the declared 11-class action schema")
    states = [list(map(float, state)) for state in initial_states]
    actions = [list(map(float, action)) for action in prior_actions]
    forecasts = []
    for action in planned_actions:
        states_window = states[-sequence_length:]
        actions_window = (actions + [list(map(float, action))])[-sequence_length:]
        padding = sequence_length - len(states_window)
        state_tensor = torch.tensor(
            [states_window + [[0.0] * len(FEATURE_NAMES) for _ in range(padding)]],
            dtype=torch.float32, device=device,
        )
        action_tensor = torch.tensor(
            [actions_window + [[0.0] * len(ACTION_CLASSES) for _ in range(padding)]],
            dtype=torch.float32, device=device,
        )
        mask_tensor = torch.tensor([[False] * len(states_window) + [True] * padding], dtype=torch.bool, device=device)
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(state_tensor, action_tensor, mask_tensor)[0])
        ensemble = torch.stack(predictions)
        mean_state = ensemble.mean(dim=0).cpu().tolist()
        variance = ensemble.var(dim=0, unbiased=len(models) > 1).mean().item()
        forecasts.append({"predicted_state": dict(zip(FEATURE_NAMES, mean_state)),
                          "mean_ensemble_variance": float(variance),
                          "action": list(action)})
        actions.append(list(map(float, action)))
        states.append(mean_state)
    return {
        "initial_state": dict(zip(FEATURE_NAMES, initial_states[-1])),
        "trajectory": forecasts,
        "warning": "Model-generated state forecasts are not causal treatment effects or clinical recommendations.",
    }


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return values.sum() / mask.sum().clamp_min(1.0)


def batch_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                  masks: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    valid = masks > 0.5
    if not torch.any(valid):
        return {"transformer_mae": None, "transformer_mse": None,
                "persistence_mae": None, "persistence_mse": None,
                "valid_target_value_count": 0}
    error = predictions[valid] - targets[valid]
    base_error = baseline[valid] - targets[valid]
    return {
        "transformer_mae": float(error.abs().mean().item()),
        "transformer_mse": float(error.square().mean().item()),
        "persistence_mae": float(base_error.abs().mean().item()),
        "persistence_mse": float(base_error.square().mean().item()),
        "valid_target_value_count": int(valid.sum().item()),
    }


def train_one(seed: int, train_examples: list[dict[str, Any]], val_examples: list[dict[str, Any]],
              args: argparse.Namespace, device: torch.device) -> tuple[TransformerDynamics, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TransformerDynamics(len(FEATURE_NAMES), len(ACTION_CLASSES), args.sequence_length,
                                args.d_model, args.heads, args.layers, args.dropout).to(device)
    train_loader = DataLoader(SequenceDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_examples), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    curve: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["states"], batch["actions"], batch["padding_mask"])
            loss = masked_smooth_l1(prediction, batch["target_state"], batch["target_mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = model(batch["states"], batch["actions"], batch["padding_mask"])
                val_losses.append(float(masked_smooth_l1(
                    prediction, batch["target_state"], batch["target_mask"]
                ).item()))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        if not math.isfinite(val_loss):
            raise FloatingPointError(
                f"Non-finite validation loss at epoch {epoch} for ensemble seed {seed}; "
                "check input values, target masks, and attention padding."
            )
        curve.append({"epoch": epoch, "train_loss": sum(train_losses) / max(1, len(train_losses)),
                      "validation_loss": val_loss})
        if val_loss < best_loss - args.minimum_improvement:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "epochs_run": len(curve), "learning_curve": curve}


def evaluate_ensemble(models: list[TransformerDynamics], examples: list[dict[str, Any]],
                      batch_size: int, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(SequenceDataset(examples), batch_size=batch_size, shuffle=False)
    prediction_chunks, variance_chunks, target_chunks, mask_chunks, base_chunks = [], [], [], [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(batch["states"], batch["actions"], batch["padding_mask"]))
        stack = torch.stack(predictions, dim=0)
        prediction_chunks.append(stack.mean(dim=0).cpu())
        variance_chunks.append(stack.var(dim=0, unbiased=True).mean(dim=1).cpu())
        target_chunks.append(batch["target_state"].cpu())
        mask_chunks.append(batch["target_mask"].cpu())
        base_chunks.append(batch["last_state"].cpu())
    if not prediction_chunks:
        return {"transition_count": 0, "metrics": None}
    predictions = torch.cat(prediction_chunks)
    variance = torch.cat(variance_chunks)
    targets = torch.cat(target_chunks)
    masks = torch.cat(mask_chunks)
    baseline = torch.cat(base_chunks)
    if not torch.isfinite(predictions).all() or not torch.isfinite(variance).all():
        raise FloatingPointError("Non-finite ensemble predictions or variance on evaluation data")
    metrics = batch_metrics(predictions, targets, masks, baseline)
    per_feature: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        valid = masks[:, index] > 0.5
        count = int(valid.sum().item())
        if count:
            model_error = predictions[valid, index] - targets[valid, index]
            baseline_error = baseline[valid, index] - targets[valid, index]
            per_feature[name] = {
                "known_targets": count,
                "transformer_mae": float(model_error.abs().mean().item()),
                "transformer_mse": float(model_error.square().mean().item()),
                "persistence_mae": float(baseline_error.abs().mean().item()),
                "persistence_mse": float(baseline_error.square().mean().item()),
            }
        else:
            per_feature[name] = {
                "known_targets": 0,
                "transformer_mae": None,
                "transformer_mse": None,
                "persistence_mae": None,
                "persistence_mse": None,
            }
    response_mask = masks[:, 2] > 0.5
    tumor_predictions: list[dict[str, Any]] = []
    if torch.any(response_mask):
        actual = targets[response_mask, 2] >= 0.5
        twin_class = predictions[response_mask, 2] >= 0.5
        persist_class = baseline[response_mask, 2] >= 0.5
        actual_positive = int(actual.sum().item())
        actual_negative = int((~actual).sum().item())
        predicted_positive = int(twin_class.sum().item())
        predicted_negative = int((~twin_class).sum().item())
        metrics["tumor_status_accuracy"] = float((twin_class == actual).float().mean().item())
        metrics["tumor_status_persistence_accuracy"] = float((persist_class == actual).float().mean().item())
        metrics["tumor_status_class_counts"] = {
            "actual_TF_Tumor_Free": actual_positive,
            "actual_WT_With_Tumor": actual_negative,
            "transformer_predicted_TF_Tumor_Free": predicted_positive,
            "transformer_predicted_WT_With_Tumor": predicted_negative,
            "persistence_predicted_TF_Tumor_Free": int(persist_class.sum().item()),
            "persistence_predicted_WT_With_Tumor": int((~persist_class).sum().item()),
        }
        selected_indices = torch.where(response_mask)[0].tolist()
        for index, true_value, model_value, base_value in zip(
            selected_indices, actual.tolist(), predictions[response_mask, 2].tolist(), baseline[response_mask, 2].tolist()
        ):
            tumor_predictions.append({
                "test_transition_index": index,
                "actual": "TF-Tumor Free" if true_value else "WT-With Tumor",
                "transformer_prediction": "TF-Tumor Free" if model_value >= 0.5 else "WT-With Tumor",
                "transformer_tumor_free_score": float(model_value),
                "persistence_prediction": "TF-Tumor Free" if base_value >= 0.5 else "WT-With Tumor",
            })
        recalls = []
        for label in (False, True):
            selected = actual == label
            if torch.any(selected):
                recalls.append(float((twin_class[selected] == actual[selected]).float().mean().item()))
        metrics["tumor_status_balanced_accuracy"] = sum(recalls) / len(recalls) if recalls else None
        metrics["tumor_status_known_targets"] = int(response_mask.sum().item())
    return {
        "transition_count": len(examples),
        "mean_ensemble_variance": float(variance.mean().item()),
        "metrics": metrics,
        "per_feature_metrics": per_feature,
        "tumor_status_predictions": tumor_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/paper_method/state_action_outcome_candidates.csv")
    parser.add_argument("--output-dir", default="data/processed/transformer_twin")
    parser.add_argument("--evaluate-checkpoint", help="Evaluate an existing ensemble checkpoint without retraining")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-train-transitions", type=int, default=100)
    parser.add_argument("--minimum-validation-transitions", type=int, default=20)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.ensemble_size < 2:
        parser.error("sequence length must be positive and ensemble size at least 2")

    input_path = Path(args.input)
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    examples, audit = make_sequences(source_rows, args.sequence_length)
    by_split = {name: [item for item in examples if item["split"] == name]
                for name in ("train", "validation", "test")}
    audit["state_features"] = list(FEATURE_NAMES)
    audit["action_classes"] = list(ACTION_CLASSES)
    audit["training_target"] = "next strictly later mapped treatment-decision state"
    audit["causal_claim"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequence_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if len(by_split["train"]) < args.minimum_train_transitions:
        raise SystemExit(f"Training stopped: only {len(by_split['train'])} train transitions; need {args.minimum_train_transitions}.")
    if len(by_split["validation"]) < args.minimum_validation_transitions:
        raise SystemExit(f"Training stopped: only {len(by_split['validation'])} validation transitions; need {args.minimum_validation_transitions}.")
    if not by_split["test"]:
        raise SystemExit("Training stopped: patient-level test split produced no transitions.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=device)
        if checkpoint.get("state_features") != list(FEATURE_NAMES):
            raise SystemExit("Checkpoint state-feature schema does not match this script")
        if checkpoint.get("action_classes") != list(ACTION_CLASSES):
            raise SystemExit("Checkpoint action schema does not match this script")
        if checkpoint.get("sequence_length") != args.sequence_length:
            raise SystemExit(
                f"Checkpoint sequence length is {checkpoint.get('sequence_length')}; "
                f"rerun with --sequence-length {checkpoint.get('sequence_length')}"
            )
        models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = TransformerDynamics(
                len(FEATURE_NAMES), len(ACTION_CLASSES), checkpoint["sequence_length"],
                checkpoint["d_model"], checkpoint["heads"], checkpoint["layers"], checkpoint["dropout"],
            ).to(device)
            model.load_state_dict(state_dict)
            models.append(model)
        test_report = evaluate_ensemble(models, by_split["test"], args.batch_size, device)
        report_path = output_dir / "transformer_training_report.json"
        result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
            "model": "Causal Transformer dynamics ensemble (paper-inspired Stage 1)",
            "device": str(device), "data_audit": audit, "training": [], "limitations": [],
        }
        result["held_out_test"] = test_report
        result["detailed_evaluation"] = {
            "checkpoint": str(Path(args.evaluate_checkpoint)),
            "retrained": False,
            "prediction_rows_include_patient_identifiers": False,
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir.resolve()), "device": str(device),
                          "retrained": False, "transitions": audit["transitions_by_split"],
                          "test": test_report}, indent=2))
        return

    models: list[TransformerDynamics] = []
    model_reports = []
    checkpoint: dict[str, Any] = {
        "state_features": list(FEATURE_NAMES), "action_classes": list(ACTION_CLASSES),
        "sequence_length": args.sequence_length, "d_model": args.d_model,
        "heads": args.heads, "layers": args.layers, "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
        "bounded_residual": 0.05,
        "model_state_dicts": [],
    }
    for member in range(args.ensemble_size):
        model, report = train_one(args.seed + member, by_split["train"], by_split["validation"], args, device)
        model_reports.append(report)
        models.append(model.to(device))
        checkpoint["model_state_dicts"].append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    torch.save(checkpoint, output_dir / "transformer_ensemble.pt")
    test_report = evaluate_ensemble(models, by_split["test"], args.batch_size, device)
    result = {
        "model": "Causal Transformer dynamics ensemble (paper-inspired Stage 1)",
        "device": str(device),
        "data_audit": audit,
        "training": model_reports,
        "held_out_test": test_report,
        "limitations": [
            "Transitions target the next recorded treatment-decision state, not a treatment-isolated immediate outcome.",
            "The patient histories contain undated treatment records, flagged in the state but not recoverable by this model.",
            "This is a next-recorded-state forecaster; it does not identify causal treatment effects.",
            "This is Stage 1 only; it does not implement the paper's adversarial outcome model or BCQ policy.",
            "A model score from this small observational cohort is not clinical validation.",
        ],
    }
    (output_dir / "transformer_training_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "device": str(device),
                      "transitions": audit["transitions_by_split"], "test": test_report}, indent=2))


if __name__ == "__main__":
    main()
