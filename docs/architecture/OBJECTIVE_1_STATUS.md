# Objective 1 — Patient Digital Twin: implementation and status

**Updated:** 2026-10-27  
**Scope:** TCGA-OV ovarian-cancer retrospective clinical JSON export  
**Status:** Initial prototype implemented and run on the supplied TCGA-OV JSON; follow-up time-axis audit and repeated baseline evaluation are complete. Local LLM comparison is prepared but awaits runtime/model availability.

**Current paper-aligned status:** The project now has a separate paper-inspired preprocessing pipeline at `../../scripts/paper_method_pipeline.py`. It builds 11-class treatment actions and candidate state/action/outcome rows, but does **not** implement the paper's Transformer ensemble, adversarial outcome model, or BCQ. The previous patient-summary and response-forecast code is an exploratory prototype, not the paper's digital twin. See `PAPER_METHOD_REPLICATION.md` for the current counts and limitations.

## Progress audit

| Objective 1 item | Status | Evidence / remaining work |
|---|---|---|
| Collect suitable healthcare datasets | Complete for the prototype dataset | `clinical.project-tcga-ov.2026-10-09.json` is now present in the project root. It contains 608 cases. MIMIC-IV is a distinct, credentialed dataset and is not present. |
| Study datasets and choose primary | Complete for prototype | TCGA-OV is selected for an ovarian-cancer long-term trajectory prototype because that is the dataset actually analyzed. It is not sufficient by itself for dense physiology or causal drug optimization. |
| Define patient state | Paper-aligned preprocessing schema specified; RL approval withheld | See `../data/patient_state_schema.md`. The paper-style pipeline constructs pre-action candidate features; the structural audit enumerates 108 field paths. It finds 364 mapped-action dated-event candidates, while the separate broad audit reports 366 under different filters; both have only 2 links in complete recorded-start histories. |
| Preprocess records | Complete for supplied export | `../../scripts/preprocess_tcga_ov.py` emitted 8,465 traceable event rows at `../../data/processed/patient_events.csv`; counts match the earlier analysis report. |
| Handle missing/noisy data | Initial policy implemented | State builder preserves absent vs explicit `Unknown`/`Not Reported`; doses retain units. There is no clinical denoising model because the export has sparse categorical events, not repeated sensor/lab signals. |
| Represent medical and treatment history | Initial implementation complete | `../../scripts/patient_twin.py` extracts diagnosis/pathology/treatment history with provenance and raw day fields. |
| Represent disease progression | Partial | Follow-up responses/progression are kept in a separate retrospective outcome section. A validated point-in-time progression state cannot yet be built because relative-day origins and availability are unverified. |
| Research suitable LLMs | Review complete; model choice deferred | See the model-selection note below. No model is installed/configured and no patient-level inference has been run. |
| Design LLM patient representation | Initial design complete | Structured state JSON plus a constrained forecast prompt and strict JSON output validation are implemented. |
| Implement initial Digital Twin | Earlier forecast prototype complete; paper digital twin not implemented | The legacy implementation is a retrospective state builder plus injected LLM forecast interface. It is not the paper's action-conditioned Transformer simulator and does not generate validated treatment counterfactuals. |
| Test patient-state generation | Run on real cohort; unit-tested on synthetic fixture | `../../data/processed/patient_states.jsonl` contains 608 generated summaries. Four unit checks pass. |
| Test trajectory prediction | Exploratory non-LLM baseline evaluated on repeated real patient holdouts | `../../scripts/evaluate_trajectory_baseline.py` evaluates next observed follow-up response over 100 seeded patient-stratified splits. LLM predictive performance remains untested. |
| Document architecture | Complete for current prototype | See `PROJECT_ARCHITECTURE.md`, this status report, schema, mapping, and decision log. |

## Dataset review and decision

**Primary prototype dataset: TCGA-OV clinical export.** The inspected export report describes 608 cases, 1,109 diagnosis records, 4,125 treatment records, and 2,036 follow-up records. The JSON is nested and has irregular event times, sparse dose/outcome fields, and no standard repeated vital-sign or laboratory series. This supports a narrow ovarian-cancer retrospective trajectory task; it does not establish adequate action-conditioned transitions for RL.

**Alternatives considered:**

- **MIMIC-IV:** rich ICU records suit short-horizon physiology, but it is not an ovarian-cancer cohort and access requires PhysioNet credentialing, training, and a data-use agreement. It would require changing the research population/question rather than silently mixing it with TCGA-OV.
- **Other TCGA/GDC cohorts or supplemental files:** could add disease-specific clinical information, but supplement schemas and coverage vary. Any addition should be explicitly mapped and kept separate until compatibility is demonstrated.
- **Genomic features:** the README lists copy-number features as a desired feature group, but the available workspace files do not include a linked genomic feature matrix. They are not represented as if present in this implementation.

**Decision:** keep TCGA-OV as the initial primary dataset for the project prototype; do not claim it is adequate for treatment optimization. The exported JSON is present and processed. Record the exact GDC release/query and license/DUA conditions with the data provenance before sharing derived artifacts.

## Follow-up time-axis audit

See `../data/FOLLOWUP_TIMELINE_AUDIT.md` and `../../data/processed/followup_timeline_audit.json`. The TCGA follow-up form specifies initial pathological diagnosis as the interval origin when intervals are submitted; GDC describes the standard field relative to the study index date. The JSON reports `index_date = Diagnosis` for 587 of 608 cases, so the trajectory analysis excludes the 21 cases without that explicit confirmation. All 1,720 populated follow-up-day values are integers between 0 and 5,481. The source arrays are not chronological in 80 of 511 cases with repeated timed non-Unknown responses; they are sorted by day before analysis. Same-day response conflicts were not observed; duplicate same-day categories are collapsed.

The TCGA Ovary Pharmaceutical Supplemental Form also defines therapy start/end intervals from initial pathological diagnosis, aligning its form-level time origin with the Ovary Followup Form. The treatment-transition audit uses cases marked `index_date = Diagnosis`, but the export contains missing and negative treatment dates and should not be treated as a complete treatment timeline. The state builder still preserves raw day values and does not yet construct decision-time states.

## Full JSON field audit and schema decision

On 2026-10-27, `../../scripts/audit_tcga_ov_fields.py` was run over all 608 case objects and nested records. It enumerated 108 distinct field paths, recording immediate-parent coverage, JSON types, observed scalar values, and array item types in `../data/dataset_field_audit.json`. This is a structural audit; it does not establish clinical semantics or data availability times.

The final state specification is `../data/patient_state_schema.md`. It distinguishes the pre-action state from the chosen treatment action, future outcomes, and reward. The treatment-transition audit is `../data/TREATMENT_TRANSITION_AUDIT.md`, with full counts in `../../data/processed/treatment_transition_audit.json`. **RL use is not approved:** 575/587 eligible patients have at least one treatment with no start day. There are 366 candidate post-completion links after excluding intervening dated treatments, but only 2 links among patients whose entire recorded treatment history has start dates. The official forms support a shared diagnosis-based time origin, while missing dates, irregular outcome lags, incomplete action details, and observational confounding prevent these counts from validating RL transitions.

## LLM selection note

The first implementation deliberately does not bind to a vendor or model. The code uses an injected callable so the same state/prompt/response checks can be used with an approved local or hosted backend.

- Medical-domain research models such as BioMistral and Meditron are candidates for structured clinical-language tasks, but medical pretraining alone does not establish reliable longitudinal forecasting or counterfactual treatment response.
- Med-PaLM 2/MedLM research supports the value of domain adaptation and human evaluation, but access, product terms, and intended-use limits must be checked before any project use. It is not assumed available.
- A general instruction model can serve as a baseline if run in an environment approved for the data. It must be evaluated against a structured non-LLM baseline and clinician-reviewed references before interpreting its outputs.

**Initial comparison candidate:** MedGemma 1.5 4B through Ollama. Its published model page describes medical/EHR text understanding; the Ollama registry lists a 3.3 GB model artifact. The benchmark adapter uses Ollama's local generation endpoint and JSON-schema constrained output. This is a practical candidate for this workstation, not a presumption that it forecasts ovarian-cancer outcomes accurately. See [MedGemma 1.5 model information](https://ollama.com/library/medgemma1.5) and [Ollama API documentation](https://ollama.readthedocs.io/en/api/). Model terms and Google’s intended-use statement must be followed; the model is not a clinical treatment tool ([intended use](https://ai.google.dev/gemma/intended_use_statement)).

**Selection rule:** compare the local medical model against the persistence and majority baselines on held-out patients for valid-output coverage and response-category prediction. Freeze model/version/prompt before evaluation. No model should create unsupported observations, reorder events, or emit treatment advice.

## Implementation and usage

The scripts use only the Python standard library.

```powershell
python scripts/preprocess_tcga_ov.py --input clinical.project-tcga-ov.2026-10-09.json --output-dir data/processed
python scripts/patient_twin.py --input clinical.project-tcga-ov.2026-10-09.json --output data/processed/patient_states.jsonl
python scripts/audit_followup_timeline.py --input clinical.project-tcga-ov.2026-10-09.json
python scripts/evaluate_trajectory_baseline.py --input clinical.project-tcga-ov.2026-10-09.json --repeats 100
python scripts/compare_llm_trajectory.py --input clinical.project-tcga-ov.2026-10-09.json --model medgemma1.5:4b
python -m unittest discover -s tests -v
```

The second command writes retrospective summaries, not leak-free decision-time states. The state builder refuses `as_of` filtering until source-time availability is verified. The forecast function accepts an LLM callable and enforces a small JSON contract; it is not connected to a model by default. The comparison script uses local Ollama and keeps patient records on the machine.

## Initial real-cohort forecast baseline

An exploratory run on the supplied 608-case export found 118 patients with at least one transition between distinct, timed, known follow-up responses. It split by patient (seed 42): 94 training patients / 98 transitions and 24 held-out patients / 24 transitions. Held-out labels were 19 `WT-With Tumor` and 5 `TF-Tumor Free`.

| Baseline | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Last-response persistence | 0.750 | 0.768 | 0.697 |
| Training majority response | 0.792 | 0.500 | 0.442 |

The majority classifier has higher raw accuracy because the held-out labels are imbalanced, but it never detects the minority class. Across 100 patient-stratified 80/20 holdouts (mean test size 24.7 transitions), persistence averaged 0.744 accuracy (split SD 0.081), 0.791 balanced accuracy (SD 0.086), and 0.684 macro F1 (SD 0.085). The training-majority model averaged 0.824 accuracy (SD 0.017), 0.500 balanced accuracy, and 0.452 macro F1 (SD 0.005). These overlapping test folds are not independent, so split standard deviations are not confidence intervals. The task predicts only the next observed response category; it does not use treatment actions or qualify as an LLM result. Full metrics are in `../../data/processed/trajectory_baseline_metrics.json`.

## Work required to close Objective 1 empirically

1. Confirm the export-to-form time-field mapping for all treatment types and resolve missing/negative dates; inspect `../data/TREATMENT_TRANSITION_AUDIT.md` before using any action-linked target.
2. Run `compare_llm_trajectory.py` using the local Ollama endpoint; compare it on the fixed holdout with persistence/majority baselines and report valid-output coverage.
3. For stronger estimates, use grouped cross-validation or patient-level bootstrap confidence intervals, then add only features available at the chosen decision time.

Until those steps are complete, Objective 1 is an implemented prototype, not an empirically validated LLM digital twin.
