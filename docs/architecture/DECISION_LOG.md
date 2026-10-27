# Architecture Decision Log

Record important choices here so that the project explains not only **what** was done, but **why**. New decisions should be appended rather than silently overwriting older ones.

## Decision record template

### ADR-XXX — Short decision title
- **Date:** YYYY-MM-DD
- **Status:** Proposed / Accepted / Superseded / Rejected
- **Context:** What problem or uncertainty prompted this decision?
- **Decision:** What are we choosing?
- **Alternatives considered:** What else could we do?
- **Rationale and evidence:** Why is this choice appropriate? Cite dataset observations or papers where possible.
- **Consequences and risks:** What does this enable, and what limitations does it introduce?
- **Validation / follow-up:** What test or evidence could confirm or overturn this decision?

---

## ADR-001 — Treat TCGA-OV as the initial preprocessing dataset
- **Date:** 2026-10-18
- **Status:** Accepted for initial exploration; final dataset choice remains open
- **Context:** The JSON currently available for analysis is a TCGA-OV ovarian-cancer cohort export, while earlier project discussions also considered MIMIC-IV.
- **Decision:** Use TCGA-OV to develop and test schema mapping and event-timeline preprocessing. Do not yet claim that it is the final dataset for the full treatment-optimization objective.
- **Alternatives considered:** Switch immediately to MIMIC-IV; combine datasets; continue with TCGA-OV as the only dataset.
- **Rationale and evidence:** The supplied JSON contains demographics, diagnosis records, nested treatment entries, follow-ups, pathology details, recurrence/progression indicators, and survival information. It is suitable for inspecting long-term oncology trajectories. However, it lacks the dense repeated vital-sign/laboratory measurements expected for an ICU physiological twin, and treatment outcomes are sparse.
- **Consequences and risks:** Enables concrete preprocessing work now, but narrows the initial population to ovarian cancer and may limit transition-level treatment learning. Cross-dataset combination is not assumed because populations, schemas, and clinical settings differ.
- **Validation / follow-up:** Count usable linked state–action–next-state examples, quantify missingness, inspect timeline validity, and compare against the requirements of the chosen clinical question.

## ADR-002 — Separate forecasting from treatment-effect estimation
- **Date:** 2026-10-18
- **Status:** Accepted
- **Context:** A model may forecast observed outcomes without identifying the effect of changing a treatment.
- **Decision:** Treat next-state prediction as an initial modelling task and action-conditioned counterfactual simulation as a stronger, separately validated claim.
- **Alternatives considered:** Treat all predictive associations as treatment effects; use an unconstrained language model to invent responses.
- **Rationale and evidence:** The dataset is observational and treatment assignment is not random. Confounding by indication and incomplete outcomes can bias treatment comparisons.
- **Consequences and risks:** Requires careful validation and conservative claims. The project must not present simulated counterfactuals as clinical truth.
- **Validation / follow-up:** Document action support, compare against held-out observed trajectories, quantify uncertainty, and design appropriate off-policy evaluation before RL.

## ADR-003 — Preserve missingness and dose units explicitly
- **Date:** 2026-10-18
- **Status:** Accepted
- **Context:** Clinical fields are inconsistently populated, and treatment doses use heterogeneous units.
- **Decision:** Keep missing values distinct from explicit labels such as `Unknown` and `Not Reported`; retain dose units alongside numeric dose values.
- **Alternatives considered:** Replace all missing-like values with one category; convert doses without agent-specific clinical conversion rules.
- **Rationale and evidence:** These values represent different data conditions, and units such as mg, mg/m², AUC, and cGy are not directly comparable.
- **Consequences and risks:** More preprocessing work and potentially larger feature spaces, but lower risk of invalid comparisons or silent data corruption.
- **Validation / follow-up:** Generate missingness reports and unit-frequency summaries; reject or flag invalid conversions.

## ADR-004 — Use diagnosis-anchored follow-up days only for eligible TCGA-OV cases
- **Date:** 2026-10-24
- **Status:** Accepted for the exploratory follow-up response benchmark
- **Context:** Follow-up records are irregular and their source array order is not always chronological. The dataset contains an `index_date` field, and merging unverified `days_to_*` fields could create invalid order.
- **Decision:** For the next-observed-follow-up response benchmark, include only cases with `index_date = Diagnosis`; order follow-ups by numeric `days_to_follow_up`; collapse duplicate same-day responses; exclude same-day conflicts and unknown/missing responses.
- **Alternatives considered:** Trust JSON array order; include all cases regardless of index metadata; merge diagnosis, treatment, and follow-up day fields immediately.
- **Rationale and evidence:** TCGA follow-up instructions define submitted intervals from initial pathological diagnosis. The supplied export has `index_date = Diagnosis` in 587/608 cases, 1,720 integer follow-up-day fields from 0 to 5,481, 80/511 repeated-response arrays stored out of time order, and no same-day conflicting known-response groups. See `docs/data/FOLLOWUP_TIMELINE_AUDIT.md`.
- **Consequences and risks:** The benchmark has only 118 eligible cases with at least two distinct, unambiguous, timed response observations (122 adjacent transitions). These are sparse observational categories at variable intervals, not regular trajectories or treatment-effect evidence. Twenty-one cases are excluded because the index reference is not explicit.
- **Validation / follow-up:** Confirm event linkages and time origin for treatment, progression, and survival fields separately; repeat evaluation with grouped cross-validation and patient-level uncertainty intervals.

## ADR-005 — Select local MedGemma 1.5 4B as first LLM comparison candidate
- **Date:** 2026-10-24
- **Status:** Proposed; benchmark adapter implemented, runtime/model unavailable
- **Context:** The forecast interface needs a reproducible model backend. No local LLM runtime was installed in the inspected environment.
- **Decision:** Prepare a zero-shot comparison adapter for MedGemma 1.5 4B via local Ollama. The adapter uses constrained JSON output and the same fixed patient-level holdout as the simple baselines.
- **Alternatives considered:** Hosted model API (requires credentials/data-handling review); larger medical LLM (higher compute requirement); leave the forecast as a mock.
- **Rationale and evidence:** MedGemma 1.5 4B is published as a medical-domain model and the Ollama registry lists a 3.3 GB artifact. Local inference avoids sending this cohort to a hosted service. Medical-domain pretraining does not establish predictive validity for this task.
- **Consequences and risks:** Actual LLM metrics remain unavailable until Ollama and model weights are installed; output coverage and invalid-response rate must be reported with accuracy. It remains an observational forecast, not counterfactual treatment simulation.
- **Validation / follow-up:** Install Ollama, pull `medgemma1.5:4b`, run `scripts/compare_llm_trajectory.py`, and compare valid-output coverage and class metrics with the fixed holdout baselines. Follow model terms and intended-use restrictions.

## Future decisions to document

## ADR-006 — Finalize a decision-time state schema, withhold RL readiness approval
- **Date:** 2026-10-27
- **Status:** Schema accepted as the project specification; RL dataset/training approval withheld
- **Context:** The initial state schema was a candidate. The full supplied TCGA-OV JSON needed a field-level audit before selecting a stable state definition.
- **Decision:** Define a pre-action decision-time state containing patient profile, temporally attributable disease status/pathology, prior treatment history, latest known follow-up, and provenance/missingness metadata. Keep the action, future response/progression/survival, and reward outside that state. Audit all JSON paths with `scripts/audit_tcga_ov_fields.py`; retain its result at `docs/data/dataset_field_audit.json`.
- **Alternatives considered:** Approve the cohort schema directly for RL; include the entire retrospective patient record; defer defining fields until an RL implementation exists.
- **Rationale and evidence:** The audit found 108 field paths in 608 cases, 4,125 treatment records, but only 2,501 treatment start days and 101 treatment outcomes. Follow-up timing is usable for a narrow exploratory response forecast, but the supplied treatment field mapping and treatment-to-outcome linkage were not yet verified at the time of this decision. Full retrospective records would leak future information into pre-action states.
- **Consequences and risks:** The project now has a stable schema contract and explicit exclusions, enabling aligned implementation and data-quality work. The schema does not itself establish that valid RL transitions, action support, causal effects, rewards, or safe policies can be learned.
- **Validation / follow-up:** Complete the readiness gates in `docs/data/patient_state_schema.md`, especially treatment time-origin verification, manually audited decision-time extraction, transition/action support analysis, domain review of actions/reward/safety, and held-out simulator/off-policy evaluation.

- Target patient population and clinical decision.
- Definition of a decision time and prediction horizon.
- Final state variables and encoding rules.
- Treatment action space and how dose/regimen is represented.
- Outcome/reward and episode termination.
- Train/validation/test split and evaluation metrics.
- Choice of structured baseline, LLM architecture, and RL algorithm.

## ADR-007 — Audit treatment-to-follow-up transition feasibility before modelling
- **Date:** 2026-10-27
- **Status:** Accepted; transition feasibility insufficient for RL approval
- **Context:** The project needed to determine whether TCGA-OV records can form temporally credible pre-action state, treatment-action, and later-response examples.
- **Decision:** Use cases marked `index_date = Diagnosis`; group same-patient, same-day nonnegative numeric treatment starts into provisional same-day treatment sets; find the first unambiguous known response on a strictly later follow-up day. Report links after treatment start and after completion separately. A stricter recorded-time subset requires all member end days, no later dated treatment start before the response, and one parent diagnosis. Also check whether every treatment start in the patient's full record is known.
- **Alternatives considered:** Link each treatment to any later response; treat the first post-start response as a fixed-horizon treatment outcome; proceed directly to RL.
- **Rationale and evidence:** TCGA pharmaceutical and follow-up forms define relevant intervals from initial pathological diagnosis. However, 1,624/4,125 treatment start days are missing and 575/587 eligible patients have at least one treatment without a start day. The audit yields 366 candidate post-completion links under a dated-event-only filter, but only 2 links among the 4 patients whose full recorded treatment-start timeline is complete. The strict-link lag is variable (median 207.5 days; 90th percentile 1,469 days).
- **Consequences and risks:** A reproducible audit is available at `docs/data/TREATMENT_TRANSITION_AUDIT.md` and `data/processed/treatment_transition_audit.json`. Counts establish retrospective association only. Undated treatments can hide intervening actions; irregular observation times and confounding prevent causal interpretation. RL remains unapproved.
- **Validation / follow-up:** Verify exported field mappings against source forms and inspect more timelines. Determine whether source-level records can resolve missing treatment dates. Choose a retrospective forecasting scope or a dataset with better treatment/action/outcome completeness before modelling.

## ADR-008 — Add paper-inspired treatment-class preprocessing without claiming model reproduction
- **Date:** 2026-10-27
- **Status:** Accepted; preprocessing adaptation implemented, RL approval withheld
- **Context:** The project chose to follow the base paper's workflow, which represents treatments as 11-class multi-hot actions, builds state/action/outcome observations, learns a digital twin, then evaluates BCQ offline. The supplied export and paper do not provide enough matching preprocessing detail to assume their rows are interchangeable.
- **Decision:** Add `scripts/paper_method_pipeline.py` to create inspectable treatment mappings, patient splits, same-day action sets, candidate pre-action features, and outcome-link quality labels. Keep the earlier event extractor, patient summary/LLM interface, and response forecast as legacy/exploratory tracks. Do not implement or claim a trained Transformer, adversarial outcome model, BCQ policy, or causal treatment effect until transition data are resolved and model evaluation is specified.
- **Alternatives considered:** Force local counts to equal the paper's; treat all post-treatment responses as rewards; move directly to RL training; discard the previous prototype scripts.
- **Rationale and evidence:** The local export contains 4,125 nested treatments versus 2,552 reported by the paper; the adaptation finds 51 action combinations versus 47. 575/587 patients have at least one treatment without a start day, and only 2 links remain in patients whose full recorded treatment-start timeline is known. The paper also reports 7,395 observations without enough described preprocessing detail to reconstruct them from the stated events.
- **Consequences and risks:** The pipeline makes the intended methodology and mapping reviewable, but its rows remain candidate observational associations. Different audit scripts have different filters: the broad transition audit reports 366 dated-event-only links, while mapped-action preprocessing reports 364. Neither count establishes causal attribution or sufficient RL support.
- **Validation / follow-up:** Reconcile paper/local event counts from source preprocessing details; manually review class mappings and candidate rows; resolve treatment-date completeness and timing; define and validate a transition dataset. Only then implement the dynamics model, reward/outcome model, and offline RL evaluation.
