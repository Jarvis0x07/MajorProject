# Project map: every script, its command and its outputs

Run all commands from the repository root. The only input is `data/raw/clinical.project-tcga-ov.2026-10-09.json`.

## Current system

| Command | Purpose | Outputs |
|---|---|---|
| `python -m digital_twin_rl all` | Every step below, in order | `data/processed/digital_twin_rl/` |
| `python -m digital_twin_rl data` | Treatment-line histories and 14-regimen action space | `patient_histories.jsonl`, `treatment_lines.csv`, `cohort_summary.json`, `regimens.json` |
| `python -m digital_twin_rl train-twin` | Outcome-model ensemble and held-out accuracy | `outcome_ensemble.pt`, `outcome_model_report.json` |
| `python -m digital_twin_rl twins` | One twin card per patient | `twins/patient_twins.jsonl` |
| `python -m digital_twin_rl twin-db` | Stored-twin database and 5-nearest-twin evaluation | `twins/twin_database.json`, `twins/neighbor_twin_report.json` |
| `python -m digital_twin_rl train-agent` | DQN and DQN+BCQ agents trained inside the twins | `agents/` |
| `python -m digital_twin_rl evaluate` | Simulation fidelity, policy comparison, recommendations, figures | `evaluation/` |
| `python -m digital_twin_rl what-if --patient <id> --line <n>` | All 14 medications for one patient at one decision | printed |
| `python -m demo` | Demo website on http://localhost:8000 | `twins/demo_created_twins.jsonl` (run-time log, not committed) |
| `python -m demo.make_sample_patients` | Demo patient files from unseen test patients | `demo/sample_patients/` |
| `python -m pytest tests -q` | All tests | — |

Package layout: `digital_twin_rl/` has `data.py` (treatment lines), `outcome_model.py` (ensemble, survival likelihood), `toxicity.py` (side-effect priors), `twin.py` (patient twin), `neighbor_twin.py` (on-the-spot twins), `record.py` (patient-file format), `environment.py` (RL environment and constraints), `policies.py` (baselines), `dqn.py` (agent), `evaluate.py` and `registry.py`. The demo backend is `demo/server.py`, and the React website is in `demo/web/src`.

## Earlier stages (kept as project history)

| Command | Purpose | Outputs |
|---|---|---|
| `python scripts/preprocess_tcga_ov.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --output-dir data/processed` | First event extraction | `patient_events.csv`, `data_quality_summary.json` |
| `python scripts/patient_twin.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --output data/processed/patient_states.jsonl` | Retrospective patient summaries (LLM-twin interface) | `patient_states.jsonl` |
| `python scripts/audit_tcga_ov_fields.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json` | Inventory of all JSON fields | `docs/data/dataset_field_audit.json` |
| `python scripts/audit_followup_timeline.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json` | Follow-up timing audit | `followup_timeline_audit.json` |
| `python scripts/audit_treatment_transitions.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json` | Treatment-to-follow-up link audit | `treatment_transition_audit.json` |
| `python scripts/evaluate_trajectory_baseline.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --repeats 100` | Response-forecast baselines | `trajectory_baseline_metrics.json` |
| `python scripts/compare_llm_trajectory.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --model medgemma1.5:4b` | LLM vs baselines (needs local Ollama) | `llm_trajectory_comparison.json` |
| `python scripts/llm_twin_rl.py ...` | LLM-twin experiment (see `--help`; needs Ollama) | `objectives_2_3/` |
| `python scripts/paper_method_pipeline.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json` | Base-paper preprocessing (11 drug classes) | `paper_method/` |
| `python scripts/transformer_digital_twin.py --input data/processed/paper_method/state_action_outcome_candidates.csv --output-dir data/processed/transformer_twin` | Transformer twin from the base paper | `transformer_twin/` |
| `python scripts/transformer_twin_rl_environment.py --output data/processed/transformer_twin/rl_environment_smoke_v2.json` | Smoke test of the Transformer environment | `transformer_twin/rl_environment_smoke_v2.json` |
| `python scripts/demo_patient_twin.py` | Demo of the early patient-summary interface with fictional data | printed |

`tests/test_patient_twin.py` covers the early patient-summary interface. `tests/test_digital_twin_rl.py` and `tests/test_demo.py` cover the current system and the demo.

## Documents

- `docs/architecture/`: design documents per stage and `DECISION_LOG.md` (ADR-001 to ADR-016).
- `docs/data/`: patient-state schema, dataset audits, dataset research notes, preprocessing validation, and the dataset analysis document.
- `docs/literature/`: the 11 survey papers; `2606.17405v1.pdf` is the base paper.
- `docs/university_forms/`: major-project report and presentation guidelines, and department forms.
