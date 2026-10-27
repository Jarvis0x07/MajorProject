# Patient Digital Twin and Offline RL for Ovarian Cancer Treatment Response

This project is being reorganized around the method in the base paper, *Treatment Response Optimized Clinical Decision Support AI System via Digital Twin Simulation*. The paper models patient history and treatment actions, learns a next-state/outcome model, then evaluates an offline RL policy inside the learned simulator.

The paper's central digital twin is a Transformer sequence model. Its LLM is used for explanations and report generation; an LLM is not the core disease-transition model.

## Current project status

The supplied TCGA-OV export contains 608 case records; 587 have `index_date = Diagnosis`. The first paper-inspired preprocessing pipeline is implemented in `scripts/paper_method_pipeline.py`. It creates:

- A 469 / 59 / 59 patient split, matching the paper's reported split.
- A treatment-event table with a transparent 11-class mapping.
- Same-day treatment decision sets with paper-style state features and one-hot action classes.
- Candidate next-response links with flags for missing dates and intervening treatments.

This is preprocessing, not a trained digital twin or RL policy. The local export has 4,125 treatment records versus 2,552 reported in the paper, and the resulting pipeline finds 51 action combinations versus 47 reported. The paper does not describe enough preprocessing detail to reconcile these counts. The pipeline finds only 2 links in patients with complete treatment-start timing across their recorded history. RL training is therefore still gated.

## Main documents

1. [`docs/architecture/PAPER_METHOD_REPLICATION.md`](docs/architecture/PAPER_METHOD_REPLICATION.md) — paper method, exact local adaptation, discrepancies, and remaining work.
2. [`docs/architecture/PROJECT_ARCHITECTURE.md`](docs/architecture/PROJECT_ARCHITECTURE.md) — project components and data flow.
3. [`docs/architecture/OBJECTIVE_1_STATUS.md`](docs/architecture/OBJECTIVE_1_STATUS.md) — current implementation and evidence.
4. [`docs/data/patient_state_schema.md`](docs/data/patient_state_schema.md) — paper-style pre-treatment state, action, next-state, and reward definition.
5. [`docs/data/TREATMENT_TRANSITION_AUDIT.md`](docs/data/TREATMENT_TRANSITION_AUDIT.md) — timing/linkage feasibility analysis.
6. [`docs/data/dataset_field_audit.json`](docs/data/dataset_field_audit.json) — structural inventory of all 108 JSON field paths.

## Run the paper-inspired preprocessing

Use the Python runtime available on your system:

```powershell
python scripts/paper_method_pipeline.py --input clinical.project-tcga-ov.2026-10-09.json
```

The output folder is `data/processed/paper_method/`:

- `pipeline_summary.json`
- `patient_splits.csv`
- `treatment_events.csv`
- `treatment_class_mapping.csv`
- `state_action_outcome_candidates.csv`

Case IDs in these files are join/split keys only and must be excluded from model features. The mapping file exposes every source agent string and its assigned class for review.

## Earlier prototype scripts

`preprocess_tcga_ov.py`, `patient_twin.py`, `evaluate_trajectory_baseline.py`, `compare_llm_trajectory.py`, and `demo_patient_twin.py` remain as the original extraction, next-response forecasting, LLM-interface, and demo work. They are useful comparisons and examples, but they are not the paper-style dynamics model or offline RL system.

## Research limits

All TCGA results are retrospective and observational. A predicted reward from a learned simulator is not a measured clinical treatment effect. Do not describe model rollouts as evidence that changing a patient's treatment would cause the predicted outcome. The project is a research prototype, not a clinical decision tool.
