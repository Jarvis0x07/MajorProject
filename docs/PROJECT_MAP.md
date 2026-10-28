# Project Map and Run Order

This document identifies the active workflow, older exploratory scripts, and generated outputs. The project root is the active repository. `Code Files/MajorProject/` is a nested archived proposal repository; its literature files are retained so that archive remains self-contained.

## Active workflow

Run from the project root:

1. **Build paper-style tables:** `py scripts/paper_method_pipeline.py --input .\clinical.project-tcga-ov.2026-10-09.json`
2. **Train/evaluate the Transformer:** `py scripts/transformer_digital_twin.py --input .\data\processed\paper_method\state_action_outcome_candidates.csv --output-dir .\data\processed\transformer_twin`
3. **Re-evaluate an existing checkpoint without training:** add `--evaluate-checkpoint .\data\processed\transformer_twin\transformer_ensemble.pt`.
4. **Smoke-test rollout plumbing:** `py scripts/transformer_twin_rl_environment.py --input .\data\processed\paper_method\state_action_outcome_candidates.csv --checkpoint .\data\processed\transformer_twin\transformer_ensemble.pt --output .\data\processed\transformer_twin\rl_environment_smoke_v2.json --episodes 5 --max-steps 3`.

The smoke test is not RL training. Its current 15 transitions returned 15 unavailable rewards because the model predicted tumor status was missing.

## Script catalog

| Script | Purpose | Status |
|---|---|---|
| `scripts/paper_method_pipeline.py` | Maps TCGA-OV treatments to 11 classes; produces splits, candidate pre-action states/actions, and linkage flags. | Active preprocessing adaptation |
| `scripts/transformer_digital_twin.py` | Builds sequences, trains/evaluates the five-member causal Transformer, or evaluates an existing checkpoint. | Active Stage 1 model |
| `scripts/transformer_twin_rl_environment.py` | Loads the Transformer and runs bounded rollouts over training-supported actions; reward is unavailable for missing tumor status. | Active integration sandbox; no RL policy |
| `scripts/audit_tcga_ov_fields.py` | Inventories JSON fields and nested structures. | Data audit |
| `scripts/audit_followup_timeline.py` | Checks follow-up dates and response ordering. | Data audit |
| `scripts/audit_treatment_transitions.py` | Audits possible treatment-to-follow-up links and timeline completeness. | Data audit |
| `scripts/preprocess_tcga_ov.py` | Flattens the source JSON into a traceable event table. | Earlier extraction track; not the paper-style action dataset |
| `scripts/patient_twin.py` | Creates retrospective state summaries and defines an injected LLM forecast interface. | Legacy prototype; not the Transformer twin |
| `scripts/evaluate_trajectory_baseline.py` | Evaluates next-observed-follow-up response baselines. | Separate forecasting task; no treatment actions |
| `scripts/compare_llm_trajectory.py` | Compares a local Ollama LLM with response baselines. | Historical LLM experiment |
| `scripts/llm_twin_rl.py` | LLM response-hypothesis environment and simple policy baselines. | Historical; not the selected twin or current RL environment |
| `scripts/demo_patient_twin.py` | Demonstrates the legacy interface with fictional data and a mock forecast. | Demo only |

`tests/test_patient_twin.py` covers the legacy patient-state/LLM interface. It does not test or validate the Transformer or the RL environment.

## Data and reports

- `clinical.project-tcga-ov.2026-10-09.json`: local source export (608 case records).
- `data/processed/paper_method/`: active paper-inspired mapped tables and split.
- `data/processed/transformer_twin/`: current model checkpoint, audit, evaluation report, and smoke outputs.
- `data/processed/transformer_twin_initial_nan_run/`: diagnostic report/audit from the failed first run; its invalid checkpoint has been removed.
- `data/processed/objectives_2_3/`: historical LLM run reports.
- `docs/data/`: state schema, field mapping, and timing audits.
- `Literature Survey/`: papers and university project forms.

The 11 exact duplicate papers inside the nested archive are intentionally retained with that archived repository. Python bytecode caches and temporary inventory files are removed and ignored.
