# Patient State Schema — TCGA-OV Prototype

**Status:** Candidate schema, not yet approved for RL training  
**Dataset:** TCGA-OV JSON export  
**Purpose:** Define what the digital twin knows about a patient at a particular decision time.

## 1. Core idea

A state is a time-specific summary of information known **by that time**. It must not contain future follow-up, future treatment, recurrence, death, or outcome information. Patient-level static characteristics can be copied forward; time-varying fields must be updated only when observed.

A conceptual state is:

`state_t = {patient_profile, disease_status_t, treatment_history_up_to_t, latest_known_followup_t, pathology, observation_metadata}`

This is a logical schema. It is not yet a tensor format or a finalized model input.

## 2. Candidate fields

| Group | Candidate fields | Type / representation | Temporal rule | Notes |
|---|---|---|---|---|
| Identity | `case_id` or pseudonymous internal patient key | String identifier | Constant | Use for joins and patient-level splits; do not use as a predictive feature. |
| Demographics | `age_at_index`, `sex_at_birth`, `race`, `ethnicity` | Numeric / categorical | Usually static | `sex_at_birth` is female in this cohort; avoid assuming it generalizes to other populations. |
| Disease | `primary_diagnosis`, `morphology`, `figo_stage`, `tumor_grade`, `icd_10_code`, `laterality` | Categorical | Diagnosis-time, with updates only if supported | Some fields are missing or have explicit `Unknown`/`Not Reported` values. |
| Diagnosis context | `classification_of_tumor`, `prior_treatment`, `method_of_diagnosis`, `year_of_diagnosis` | Categorical / numeric | Event-linked | Distinguish initial primary diagnosis from recurrence, metastasis, and progression records. |
| Treatment history | `treatment_type`, `therapeutic_agents`, `treatment_intent_type`, `route_of_administration`, `number_of_cycles` | Categorical / numeric | Include only treatments occurring by time t | Treatment records are nested under diagnoses; preserve their parent diagnosis link. |
| Dose | `treatment_dose`, `treatment_dose_units`, `prescribed_dose`, prescribed-dose units | Numeric + categorical unit | Event-linked | Never compare numeric dose without agent, unit, route, and context. Sparse values. |
| Disease response | `disease_response`, `timepoint_category` | Categorical | Latest observation at or before t | Do not expose future response labels to the state. |
| Progression / recurrence | `progression_or_recurrence`, `progression_or_recurrence_type`, `days_to_recurrence`, `days_to_progression` | Categorical / relative day | Event/outcome label | These are potential targets, not automatically valid current-state features. Use only after the event is known. |
| Function | `ecog_performance_status`, `karnofsky_performance_status` | Ordinal / numeric | Latest known value at or before t | Sparse; retain missingness indicators. |
| Survival | `vital_status`, `days_to_death`, `days_to_last_follow_up` | Categorical / relative day | Outcome / censoring | Often leakage if used as input before death or end of follow-up. Prefer as target or censoring metadata. |
| Pathology | `vascular_invasion_present`, `lymphatic_invasion_present`, `residual_tumor_measurement` | Categorical | Pathology time | Include only if available by decision time. |
| Time metadata | `event_day`, `event_type`, `time_since_previous_event`, `source_record_id` | Numeric / categorical | Each event | Preserve original relative-day values and document their reference point. |
| Missingness | Per-field missingness indicators and original value status | Boolean / categorical | Per observation | Distinguish absent field, null, `Unknown`, `Not Reported`, and explicit negative values. |

## 3. Fields that should usually be targets or evaluation metadata, not state inputs

- Future `disease_response` or future follow-up records.
- Future recurrence/progression events and their event times.
- `vital_status = Dead` and `days_to_death` before the death event is known.
- Treatment outcome fields recorded after a treatment has been administered.
- `updated_datetime`, database record IDs, and other administrative metadata that may leak dataset curation rather than clinical state.

Whether a field belongs in the state depends on the exact decision time. A variable may be a target at one time and a valid observed state feature later.

## 4. Event and transition definitions (provisional)

- **Patient:** a case record identified by the dataset's case identifier.
- **Event:** a diagnosis, treatment, or follow-up record with a relative-day field or otherwise documented temporal position.
- **Decision time:** not yet fixed. It must be chosen based on the clinical question and available temporal resolution.
- **Action:** a treatment regimen/action represented using the treatment record. The exact action space is not yet fixed.
- **Next state:** the next defensible observed patient state after the action, with elapsed time explicitly retained.
- **Reward/outcome:** not defined yet. It must not be improvised from incomplete response labels without clinical and methodological justification.

## 5. Important data limitations

1. The cohort contains 608 case records, but nested diagnosis, treatment, and follow-up records have different coverage.
2. Treatment outcome is sparse relative to treatment-event count.
3. Follow-up times are irregular and fields are not consistently present.
4. The data is observational. A treatment followed by an outcome does not prove that the treatment caused that outcome.
5. TCGA-OV does not provide dense repeated vital signs or laboratory trajectories in this JSON export.
6. Relative-day fields may have different reference points. Do not sort or merge them into one timeline until their semantics are verified from the data dictionary/source documentation.

## 6. Encoding policy (initial)

- Keep raw categorical labels before encoding.
- Keep numeric value and unit in separate columns.
- Do not silently convert `Unknown` or `Not Reported` to a negative finding.
- Add explicit missingness flags where useful.
- Keep source IDs and raw time fields for traceability, but exclude identifiers from model features.
- Fit scalers, imputers, and encoders on the training split only.
- Split by patient before fitting preprocessing transforms to avoid leakage.

## 7. Approval checklist

- [ ] Choose one clinical question and target population.
- [ ] Verify the reference point of every relative-day field.
- [ ] Confirm which fields are available at the proposed decision time.
- [ ] Quantify missingness and treatment-outcome linkage.
- [ ] Define an initial prediction target and horizon.
- [ ] Validate chronological extraction on manually inspected patient examples.
- [ ] Approve state and action spaces before RL formulation.
