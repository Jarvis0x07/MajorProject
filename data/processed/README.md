# Processed Data and Model Outputs

These files are generated from the TCGA-OV source export or from project scripts. The source export is at the project root; rerun the indicated script to regenerate an output when needed.

## Current paper-inspired pipeline

- `paper_method/`: treatment mapping, patient splits, decision-state/action candidates, and pipeline summary. Generate with `scripts/paper_method_pipeline.py`.
- `transformer_twin/`: current five-member Transformer checkpoint, sequence audit, training/evaluation report, and environment smoke reports. Generate the model files with `scripts/transformer_digital_twin.py`; run the integration sandbox with `scripts/transformer_twin_rl_environment.py`.
- `transformer_twin_initial_nan_run/`: first run with invalid `NaN` metrics, retained as a report/audit debugging record; its model checkpoint was removed because it was invalid and is not to be loaded.

## Earlier exploratory outputs

- `objectives_2_3/`: historical LLM-simulator experiment reports; not the selected digital twin.
- `patient_events.csv`, `patient_states.jsonl`, `data_quality_summary.json`: outputs of the initial event extraction and retrospective patient-summary work.
- `followup_timeline_audit.json`, `treatment_transition_audit.json`, and `trajectory_baseline_metrics.json`: separate data-quality audits and response-forecast baselines. Their filters/targets differ; compare counts only with their associated documentation.

The current environment smoke result is `transformer_twin/rl_environment_smoke_v2.json`: 15 transitions completed, with all rewards unavailable because tumor status was predicted missing. It is an integration check, not an RL result.
