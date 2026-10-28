# Transformer Patient Digital Twin and Offline RL for Ovarian Cancer

This project follows the base paper's core method: a causal Transformer ensemble models patient-state transitions from state/action histories; an outcome/reward model and BCQ policy are later stages. The LLM is reserved for explanation/reporting, not used as the disease-transition model.

## Current status

The TCGA-OV export has 608 case records; 587 have `index_date = Diagnosis`. The paper-inspired preprocessing and 11-class treatment mapping are in `scripts/paper_method_pipeline.py`. The local export has 4,125 treatment records versus 2,552 reported in the paper, and 51 action combinations versus 47. Only two candidate response links remain among patients with complete recorded treatment-start histories.

The Transformer dynamics ensemble has been trained and evaluated. It forecasts the next recorded treatment-decision state, not a treatment-isolated causal effect. Aggregate error is lower than persistence, but stable features favor persistence and tumor status has only 13 known targets, all `WT-With Tumor`. A rollout-environment smoke test succeeded; all 15 rewards were correctly unavailable because tumor status was missing. No outcome/reward model or BCQ/RL policy has been trained. RL and clinical use are not approved.

The earlier LLM-based experiment remains as a comparison artifact, not the selected twin. Its full held-out report is `data/processed/objectives_2_3/full_holdout_comparison.json`: the LLM predicted only `WT-With Tumor`, with 73.3% accuracy on valid rows versus 74.2% for the majority baseline, and 0% recall for tumor-free cases.

## Read the docs

1. [`docs/architecture/PAPER_METHOD_REPLICATION.md`](docs/architecture/PAPER_METHOD_REPLICATION.md) — what the paper did and where local results differ.
2. [`docs/architecture/OBJECTIVES_2_3.md`](docs/architecture/OBJECTIVES_2_3.md) — Transformer twin, sequence target, MDP, and current status.
3. [`docs/data/patient_state_schema.md`](docs/data/patient_state_schema.md) — state/action/target separation and field mapping.
4. [`docs/data/TREATMENT_TRANSITION_AUDIT.md`](docs/data/TREATMENT_TRANSITION_AUDIT.md) — treatment timing and outcome-link limitations.
5. [`docs/architecture/DECISION_LOG.md`](docs/architecture/DECISION_LOG.md) — decisions, including the switch from the LLM twin experiment to the paper's Transformer approach.
6. [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) — script catalog, run order, and output locations.

## Folder map

- `scripts/` — preprocessing, audits, Transformer training, environment smoke runner, and retained exploratory scripts.
- `docs/` — current architecture, dataset schema/audits, and method adaptation.
- `data/processed/` — generated tables, model checkpoints, and run reports; see its [`README.md`](data/processed/README.md).
- `Literature Survey/` — source papers and project forms.
- `Code Files/MajorProject/` — archived nested proposal repository; the project-root README and `docs/` are the current source of truth.

## Prepare the local paper-style data

From this project directory:

```powershell
python scripts/paper_method_pipeline.py --input .\clinical.project-tcga-ov.2026-10-09.json
```

Outputs are written to `data/processed/paper_method/`, including `state_action_outcome_candidates.csv`. Patient identifiers in these derived files are join/split keys and must not be model inputs.

## Train and evaluate the Transformer dynamics ensemble

Install PyTorch into the same Python environment used to run the project:

```powershell
python -m pip install -r requirements-transformer.txt
```

Then run:

```powershell
python scripts/transformer_digital_twin.py `
  --input .\data\processed\paper_method\state_action_outcome_candidates.csv `
  --output-dir .\data\processed\transformer_twin
```

By default, it trains five independent Transformer models and writes a sequence audit, model checkpoint, and held-out report. Do not interpret forecasts as effects of changing treatment: target states are the next recorded treatment-decision states, treatment histories are incomplete, and the paper's preprocessing is not fully specified.

## Smoke-test the Transformer as an environment

After training, run a few random rollouts using training-supported action combinations:

```powershell
py scripts/transformer_twin_rl_environment.py `
  --input .\data\processed\paper_method\state_action_outcome_candidates.csv `
  --checkpoint .\data\processed\transformer_twin\transformer_ensemble.pt `
  --output .\data\processed\transformer_twin\rl_environment_smoke_v2.json `
  --episodes 5 `
  --max-steps 3
```

This checks integration only. The recorded v2 smoke run allowed 25 training action combinations (minimum support 5), completed 15 steps, and had 15 unavailable rewards because tumor status remained missing. When tumor status is present, the placeholder reward is the model-predicted tumor-free state component, not a validated outcome or treatment utility. It does not train an RL policy; policy learning remains gated by outcome-data quality and independent evaluation.

## LLM comparison (historical experiment)

`scripts/llm_twin_rl.py` is the earlier local LLM hypothesis generator and baseline comparison. It is retained to document why it was not selected as the digital twin. It is not the current project direction and does not train an RL policy.

## Research limits

The project is retrospective and observational. A predicted reward is not a measured clinical treatment effect. Toxicity, cost, and quality-of-life outcomes are not adequately represented in this export. The Transformer, outcome model, and RL policy require separate evaluation, and this project is not a clinical decision tool.
