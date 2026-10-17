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

## Future decisions to document

- Target patient population and clinical decision.
- Definition of a decision time and prediction horizon.
- Final state variables and encoding rules.
- Treatment action space and how dose/regimen is represented.
- Outcome/reward and episode termination.
- Train/validation/test split and evaluation metrics.
- Choice of structured baseline, LLM architecture, and RL algorithm.
