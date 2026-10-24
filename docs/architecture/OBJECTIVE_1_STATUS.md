# Objective 1 — Patient Digital Twin: implementation and status

**Updated:** 2026-10-24  
**Scope:** TCGA-OV ovarian-cancer retrospective clinical JSON export  
**Status:** Initial prototype implemented; empirical/model-backed validation is pending.

## Progress audit

| Objective 1 item | Status | Evidence / remaining work |
|---|---|---|
| Collect suitable healthcare datasets | Partial | A 608-case TCGA-OV export is described in `TCGA_OV_JSON_Dataset_Analysis.docx`, but the source JSON is not in this workspace. MIMIC-IV is a distinct, credentialed dataset and is not present. |
| Study datasets and choose primary | Complete for prototype | TCGA-OV is selected for an ovarian-cancer long-term trajectory prototype because that is the dataset actually analyzed. It is not sufficient by itself for dense physiology or causal drug optimization. |
| Define patient state | Candidate schema complete | See `../data/patient_state_schema.md` and `../data/dataset_to_state_mapping.csv`; not approved for RL. |
| Preprocess records | Extraction implementation complete | `../../scripts/preprocess_tcga_ov.py` emits traceable event rows. Source JSON is needed to rerun on the project dataset. |
| Handle missing/noisy data | Initial policy implemented | State builder preserves absent vs explicit `Unknown`/`Not Reported`; doses retain units. There is no clinical denoising model because the export has sparse categorical events, not repeated sensor/lab signals. |
| Represent medical and treatment history | Initial implementation complete | `../../scripts/patient_twin.py` extracts diagnosis/pathology/treatment history with provenance and raw day fields. |
| Represent disease progression | Partial | Follow-up responses/progression are kept in a separate retrospective outcome section. A validated point-in-time progression state cannot yet be built because relative-day origins and availability are unverified. |
| Research suitable LLMs | Review complete; model choice deferred | See the model-selection note below. No model is installed/configured and no patient-level inference has been run. |
| Design LLM patient representation | Initial design complete | Structured state JSON plus a constrained forecast prompt and strict JSON output validation are implemented. |
| Implement initial Digital Twin | Prototype complete | The implementation is a state builder plus an injected LLM forecast interface. It is not a validated simulator and does not generate treatment counterfactuals. |
| Test patient-state generation | Unit-tested on synthetic fixture | `tests/test_patient_twin.py`; real cohort run awaits the missing source JSON. |
| Test trajectory prediction | Interface/contract-tested with a deterministic fake model | This verifies prompt construction, parsing, and validation only. Predictive performance requires an actual model and held-out cohort. |
| Document architecture | Complete for current prototype | See `PROJECT_ARCHITECTURE.md`, this status report, schema, mapping, and decision log. |

## Dataset review and decision

**Primary prototype dataset: TCGA-OV clinical export.** The inspected export report describes 608 cases, 1,109 diagnosis records, 4,125 treatment records, and 2,036 follow-up records. The JSON is nested and has irregular event times, sparse dose/outcome fields, and no standard repeated vital-sign or laboratory series. This supports a narrow ovarian-cancer retrospective trajectory task; it does not establish adequate action-conditioned transitions for RL.

**Alternatives considered:**

- **MIMIC-IV:** rich ICU records suit short-horizon physiology, but it is not an ovarian-cancer cohort and access requires PhysioNet credentialing, training, and a data-use agreement. It would require changing the research population/question rather than silently mixing it with TCGA-OV.
- **Other TCGA/GDC cohorts or supplemental files:** could add disease-specific clinical information, but supplement schemas and coverage vary. Any addition should be explicitly mapped and kept separate until compatibility is demonstrated.
- **Genomic features:** the README lists copy-number features as a desired feature group, but the available workspace files do not include a linked genomic feature matrix. They are not represented as if present in this implementation.

**Decision:** keep TCGA-OV as the initial primary dataset for the project prototype; do not claim it is adequate for treatment optimization. The exported source JSON must be restored locally before data-level validation. Record the exact GDC release/query and license/DUA conditions with the data when it is restored.

## LLM selection note

The first implementation deliberately does not bind to a vendor or model. The code uses an injected callable so the same state/prompt/response checks can be used with an approved local or hosted backend.

- Medical-domain research models such as BioMistral and Meditron are candidates for structured clinical-language tasks, but medical pretraining alone does not establish reliable longitudinal forecasting or counterfactual treatment response.
- Med-PaLM 2/MedLM research supports the value of domain adaptation and human evaluation, but access, product terms, and intended-use limits must be checked before any project use. It is not assumed available.
- A general instruction model can serve as a baseline if run in an environment approved for the data. It must be evaluated against a structured non-LLM baseline and clinician-reviewed references before interpreting its outputs.

**Selection rule:** compare at least one locally runnable biomedical model and one general instruction model on held-out patient cases for schema adherence, evidence attribution, response-category prediction, calibration, and error analysis. Freeze model/version/prompt before evaluation. No model should be allowed to create unsupported lab values, reorder events with unverified dates, or emit treatment advice.

## Implementation and usage

The scripts use only the Python standard library.

```powershell
python scripts/preprocess_tcga_ov.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --output-dir data/processed
python scripts/patient_twin.py --input data/raw/clinical.project-tcga-ov.2026-10-09.json --output data/processed/patient_states.jsonl
python -m unittest discover -s tests -v
```

The second command writes retrospective summaries, not leak-free decision-time states. The state builder refuses `as_of` filtering until the relative-day reference points have been verified. The forecast function accepts an LLM callable and enforces a small JSON contract; it is not connected to a model by default.

## Work required to close Objective 1 empirically

1. Put the licensed/authorized source JSON under the documented local raw-data path (do not commit it without permission).
2. Verify `days_to_diagnosis`, `days_to_treatment_*`, `days_to_follow_up`, progression, and survival time origins against source documentation; manually audit linked cases.
3. Run extraction and state generation on the actual JSON; compare counts/missingness to the existing analysis report and investigate discrepancies.
4. Define one target and forecast horizon, then build patient-level train/validation/test partitions with point-in-time feature availability.
5. Connect a selected LLM backend, compare it to a structured baseline, and report held-out predictive metrics, calibration, subgroup/error analysis, and limitations.

Until those steps are complete, Objective 1 is an implemented prototype, not an empirically validated LLM digital twin.
