# Legacy event-extractor run and paper-style preprocessing status

This note records the earlier `preprocess_tcga_ov.py` extraction run. Those counts are still useful as a raw parsing inventory; they are **not** the paper-style decision/action counts.

The extractor was executed against the available source JSON during package creation.

- Input cases: 608
- Event rows: 8465
- Event counts: {"diagnosis": 1109, "follow_up": 2036, "pathology": 587, "patient_baseline": 608, "treatment": 4125}
- Cases without demographic object: 21
- Cases without diagnoses array: 21
- Cases without follow-ups array: 24
- Treatments without start-day field: 1624
- Follow-ups without follow-up-day field: 316

The output event table is not included here because it is derived patient-level data. Re-run the script locally after placing the source file in your permitted local data directory.

**Validation boundary:** successful execution verifies parsing and extraction mechanics only. It does not validate the clinical meaning of event order, prove causal treatment effects, or establish that the data is sufficient for reinforcement learning.

## Current paper-inspired pipeline

The separate `scripts/paper_method_pipeline.py` emits mapped therapy classes, patient-level splits, and candidate same-day state/action/outcome rows under `data/processed/paper_method/`. The latest summary reports:

- 587 diagnosis-indexed cases; patient split 469/59/59.
- 4,125 raw nested treatment records, of which 2,501 have numeric start days and 2,188 have numeric end days.
- 1,453 same-day action sets with at least one mapped class; 1,449 have all treatment members mapped.
- 51 local action combinations versus the paper's reported 47.
- 364 mapped-action dated-event-only response candidates, but only 2 links among patients whose full recorded treatment-start history is complete.

These are reproducible preprocessing outputs, not a trained model or RL-ready transitions. The earlier treatment audit's 366 links use a different, broader event/action filter; see `docs/data/TREATMENT_TRANSITION_AUDIT.md` and `docs/architecture/PAPER_METHOD_REPLICATION.md`.
