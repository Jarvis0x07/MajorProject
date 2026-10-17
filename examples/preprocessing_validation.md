# Preprocessing script validation

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
