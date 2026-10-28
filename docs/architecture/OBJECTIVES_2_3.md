# Objectives 2 and 3: Transformer Digital Twin and RL Formulation

**Updated:** 2026-10-29  
**Status:** Transformer ensemble trained and evaluated. The held-out aggregate error is lower than persistence, but the improvement is feature-dependent; tumor status has only 13 known targets, all `WT-With Tumor`, so tumor-free response is untested. An experimental Transformer-backed rollout environment is now implemented for integration smoke runs; it does not train an RL policy.  
**Scope:** Retrospective TCGA-OV research only; not for clinical use.

## What the base paper actually does

The paper's patient twin is **not an LLM**. It uses five causal Transformer models. Each model consumes a sequence of patient states aligned with treatment actions and predicts the next state. A bounded residual update limits each state change; the ensemble mean is the forecast and ensemble variance supplies an uncertainty signal.

The paper's full system then adds two more stages:

1. An adversarially regularized treatment-outcome/reward model estimates outcome from state and action. Its authors explicitly state that this does not eliminate unmeasured confounding.
2. BCQ offline RL restricts candidate actions to those supported by the historical behavior model and chooses the highest-valued supported action.

The paper uses an LLM only as a tool-augmented explanation/reporting layer that cites model/tool outputs. It is not the disease dynamics model. It also describes online learning and uncertainty-based clinician queries, which are later stages beyond this project's current implementation.

## Our Transformer twin implementation

The model code is [`scripts/transformer_digital_twin.py`](../../scripts/transformer_digital_twin.py). It implements the paper-inspired Stage 1 architecture:

- causal Transformer encoder over recent state and 11-class action sequences;
- padding mask for unequal patient-history lengths;
- five independently initialized ensemble members by default;
- bounded residual state update (`0.05 * tanh(...)`) and clipping to `[0,1]`, following the paper's stated equation;
- masked Smooth L1 loss, AdamW, gradient clipping, validation early stopping;
- ensemble mean and variance, held-out state-transition metrics, and persistence baseline;
- `simulate_trajectory(...)` helper for bounded multi-step forecasts under a supplied sequence of candidate actions.

The state encoding contains normalized age, tumor response, grade, stage, cumulative dated drug count, prior radiation, ECOG, Karnofsky, missingness indicators, and the undated-treatment-history flag. Constant sex is omitted from the model vector because it has no within-cohort variation. Actions are the paper's 11 multi-hot treatment classes.

### What the current training target means

The script forms adjacent same-patient, strictly later, fully mapped treatment-decision rows. Given the history through one treatment decision and its observed action, it predicts the state recorded before the next treatment decision. This is a next-recorded-decision **forecast**. It is not a verified immediate post-treatment state and does not isolate the effect of that action. The model carries the missing-history flag, but cannot reconstruct untimed treatments.

This target lets us build and evaluate the sequence-model machinery without fabricating counterfactual labels. It is a local adaptation, not an exact reconstruction of the paper's undocumented 7,395 outcome-observation preprocessing.

### Detailed held-out evaluation of the saved ensemble

To evaluate the already-trained checkpoint without training again, run:

```powershell
py scripts/transformer_digital_twin.py `
  --input .\data\processed\paper_method\state_action_outcome_candidates.csv `
  --output-dir .\data\processed\transformer_twin `
  --evaluate-checkpoint .\data\processed\transformer_twin\transformer_ensemble.pt
```

The report now includes per-feature MAE/MSE against persistence, known-target counts, tumor-status class counts, and anonymized tumor-status predictions. The summary is updated in `transformer_training_report.json`; no model retraining occurs.

## Local transition availability

Using fully mapped action rows in `state_action_outcome_candidates.csv`, the current patient-level split contains:

| Split | Decision rows | Adjacent later-decision transitions | Patients contributing transitions |
|---|---:|---:|---:|
| Train | 1,146 | 739 | 262 |
| Validation | 154 | 101 | 34 |
| Test | 149 | 102 | 36 |

These are adjacent decision-state forecasts, not the separate treatment-to-response links. The patient IDs remain partitioned by the existing 469/59/59 split. Many of these histories include undated treatments. Only 2 candidate response links survive the full recorded treatment-start completeness check, so the data does **not** support a trustworthy treatment-effect model or RL policy.

## Install and run

PyTorch is required. Install `requirements-transformer.txt` into the same Python environment you use to run the script. Prepare the paper-inspired CSV if it is absent, then train/evaluate the Stage 1 ensemble:

```powershell
python scripts/paper_method_pipeline.py --input .\clinical.project-tcga-ov.2026-10-09.json
python -m pip install -r requirements-transformer.txt
python scripts/transformer_digital_twin.py `
  --input .\data\processed\paper_method\state_action_outcome_candidates.csv `
  --output-dir .\data\processed\transformer_twin
```

Default settings use sequence length 8, five ensemble members, 64-dimensional hidden representations, up to 50 epochs, and early stopping. Training fails closed if the train/validation transition counts fall below the configured minimums. Outputs are:

- `sequence_audit.json`: split, patient, transition, and incomplete-history counts;
- `transformer_ensemble.pt`: five trained model state dictionaries plus feature/action schema;
- `transformer_training_report.json`: learning curves and held-out metrics.

The current run produced a finite report at `data/processed/transformer_twin/transformer_training_report.json`. The previous [LLM full-holdout comparison](../../data/processed/objectives_2_3/full_holdout_comparison.json) remains a historical comparison artifact only.

## Objective 3 MDP formulation

| Component | Current definition | Limitation |
|---|---|---|
| State | Paper-style pre-action vector with missingness flags | Some diagnosis-field availability times and prior actions are uncertain |
| Action | Observed multi-hot combinations of 11 therapy classes | No dose, contraindication, or personalized eligibility model |
| Transition | Transformer ensemble forecasts the next recorded decision state | Not immediate post-treatment; no causal identification |
| Reward proxy | 1 for tumor-free response, 0 for with-tumor response | Sparse/irregular; not yet learned or validated by the Transformer |
| Discount | Paper reports `γ = 0.99`; any engineering default must be stated | Irregular time between recorded decisions |
| Episode | Event-based sequence, terminated by configured rollout horizon | Not a fixed calendar schedule or survival endpoint |
| Support constraint | Restrict candidate actions to combinations observed in training | Observed frequency is not proof of clinical appropriateness |
| Safety | No action outside the historical class space; abstain on unsupported inputs | This is not clinical safety screening; no toxicity/contraindication data |

The data does not contain adequate toxicity, cost, or quality-of-life targets. We therefore do not invent penalties or claim multi-objective optimization. The earlier random/historical-frequency baselines can be used as behavior comparisons, but they are not clinical guidelines.

## Transformer rollout environment (integration sandbox)

[`scripts/transformer_twin_rl_environment.py`](../../scripts/transformer_twin_rl_environment.py) loads the trained ensemble, starts episodes from training-split patient histories, and permits only action combinations observed at least five times in the training split by default. Its Gym-like `reset()`/`step()` interface predicts the next recorded-decision state. It returns no reward when predicted tumor status is missing; otherwise it returns the predicted tumor-free state component as a **placeholder proxy reward**. The command-line smoke runner uses uniform random supported actions and writes a rollout report:

```powershell
py scripts/transformer_twin_rl_environment.py `
  --input .\data\processed\paper_method\state_action_outcome_candidates.csv `
  --checkpoint .\data\processed\transformer_twin\transformer_ensemble.pt `
  --output .\data\processed\transformer_twin\rl_environment_smoke.json `
  --episodes 5 `
  --max-steps 3
```

This verifies model/environment plumbing only. The previous smoke output assigned positive proxy rewards while the predicted tumor-status missingness remained near 1.0; the environment now treats that reward as unavailable. Even a non-missing proxy is not a measured outcome, action frequency does not imply safety, and optimization inside this same model could exploit its errors. **No RL agent is trained or evaluated by this script.** The paper's next method stage is a separately trained outcome/reward model; the local high-integrity action-response links are too sparse to validate one, so policy learning remains gated.

## Status: what is and is not done

Implemented:

- Transformer ensemble architecture and paper-style bounded residual;
- sequence construction using the patient-level train/validation/test split;
- training loop, masked loss, early stopping, ensemble uncertainty, persistence metrics;
- multi-step prediction helper and explicit state/action/reward/termination definitions.
- an experimental Transformer-backed rollout environment with training-supported action combinations and an explicitly unvalidated proxy reward.

Not done:

- stable, clinically relevant validation of the Transformer, especially on tumor-free targets;
- the paper's adversarial treatment-outcome model and BCQ policy;
- any RL policy training or meaningful offline policy evaluation;
- causal treatment-effect estimation or independent off-policy evaluation;
- dose, toxicity, cost, wellbeing, or clinical contraindication models;
- clinician-reviewed safety validation or any clinical use.

Next milestone: run the environment smoke command to verify checkpoint-to-environment integration. Treat it as a software check only. Before training RL, obtain or construct a sufficiently complete, temporally credible treatment/outcome dataset and validate an outcome model and policy evaluation separately. Even a good next-state forecast would not by itself validate counterfactual treatment choices.
