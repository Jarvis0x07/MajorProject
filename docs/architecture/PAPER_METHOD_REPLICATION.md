# Base-Paper Method: Local Replication Plan and Dataset Fit

**Base paper:** Xinyu Qin et al., *Treatment Response Optimized Clinical Decision Support AI System via Digital Twin Simulation*, arXiv:2606.17405v1 (2026). The paper PDF is in `Literature Survey/2606.17405v1.pdf`; the [arXiv record](https://arxiv.org/abs/2606.17405) provides the public version.

## What the paper reports

The paper describes an offline-trained clinical decision support system with three learned parts:

1. **Dynamics model / digital twin:** a Transformer ensemble predicts the next patient state from the patient's state history and treatment actions. It bounds each predicted update and uses ensemble disagreement as an uncertainty signal.
2. **Treatment outcome / reward model:** estimates an immediate outcome from state and action. The paper adds an adversarial action-prediction objective intended to reduce reliance on treatment-assignment patterns; it explicitly does not eliminate unmeasured confounding.
3. **Offline RL policy:** BCQ restricts recommendations to actions supported by the historical behavior model. It selects the highest-value action within that supported set.

The paper also describes uncertainty-based clinician querying and online updates. Those deployment components are later-stage work and are not implemented in this repository.

**Project decision:** follow the paper's Stage 1 Transformer ensemble, using the paper-inspired cohort, state features, multi-hot actions, and patient-level split. The Stage 1 implementation is [`transformer_digital_twin.py`](../../scripts/transformer_digital_twin.py), documented in [`OBJECTIVES_2_3.md`](OBJECTIVES_2_3.md). The LLM-based prototype remains only as a historical comparison; the paper uses its LLM for explanations/report generation, not disease dynamics. The local Transformer is an adaptation and does not claim to reproduce the paper's unreported preprocessing details or results.

### Paper's TCGA-OV setup as reported

- 587 ovarian-cancer patients and 2,552 treatment events.
- Drug names normalized into 11 classes: platinum, taxanes, anthracyclines, antimetabolites, topoisomerase inhibitors, alkylating agents, anti-angiogenics, hormonal agents, vinca alkaloids, radiation, and other.
- 47 treatment combinations, represented as multi-hot actions.
- State features: age, gender, tumor status, grade, stage, cumulative drug count, radiation history, ECOG, Karnofsky, and reserved features.
- Binary reward: 1 for a tumor-free transition and 0 otherwise; 27.5% positives reported.
- Patient split: 469 train, 59 validation, 59 test.
- Five algorithm comparisons: BCQ, DQN, Double DQN, NFQ, and CQL. Policy returns are evaluated in the learned digital twin, not by assigning alternative treatments to real patients.

The paper reports 7,395 treatment-outcome observations but does not give enough detail to reconstruct that count from its described 2,552 treatment events. Its event-cleaning rules, handling of missing treatment dates, definition of state/action rows, dose/regimen handling, and exact target horizon are not fully specified. This local project therefore implements a documented adaptation rather than claiming an exact reproduction.

## What this repository now builds

Run `scripts/paper_method_pipeline.py` on the checked-in source export. It writes to `data/processed/paper_method/`:

- `patient_splits.csv`: patient-level 469/59/59 split, seed 42.
- `treatment_events.csv`: all eligible treatment records with raw agent, date, dose/unit, and mapped action class.
- `treatment_class_mapping.csv`: every observed agent string and the local mapping used.
- `state_action_outcome_candidates.csv`: same-patient, same-day treatment sets with state fields available before the action, action multi-hot fields, a later-response target when available, and linkage-quality flags.
- `pipeline_summary.json`: cohort, mapping, action-combination, outcome, and reproduction counts.

The pre-action state uses the paper's reported features where the JSON supports them:

- Age and sex at birth (sex is female for all 587 recorded patients, so it carries no within-cohort variation).
- The primary diagnosis's grade and FIGO stage.
- The latest unambiguous tumor-response status, ECOG, and Karnofsky values strictly before the treatment day.
- A count of earlier **dated treatment records**, plus an earlier-radiation indicator. Missing treatment starts mean this count can understate true history; a missingness flag is included.

The action is a multi-hot vector over the 11 paper classes. Records for surgery remain in the event output but are excluded from the paper's therapy-class action space because surgery is not one of its stated classes. Known drug strings that do not match an explicit mapping fall into `other`; missing/unknown agents stay unmapped. This is visible in the generated mapping table.

The response target is the first unambiguous known follow-up response strictly after all available end days for that same-day treatment set. Reward is 1 for `TF-Tumor Free`, 0 for another known response, and missing when no qualifying response is recorded. The table retains broader post-start candidates separately from completed-treatment links.

## Local results and discrepancy with the paper

The pipeline was run on `clinical.project-tcga-ov.2026-10-09.json`:

| Check | Local result | Paper report |
|---|---:|---:|
| Cases used | 587 | 587 |
| Treatment records | 4,125 | 2,552 |
| Numeric treatment start days | 2,501 | Not specified |
| Numeric treatment end days | 2,188 | Not specified |
| Same-day action sets with at least one mapped action class | 1,453 | Not specified |
| Unique local class combinations | 51 | 47 |
| Complete patient histories with a dated start for every treatment | 4 patients | Not specified |
| Post-completion links within those complete histories | 2 | Not specified |

The cohort size agrees, but event extraction does not. This export has 575/587 eligible patients with at least one treatment record missing a start day. The current action grouping is provisional and local; it should not be silently tuned just to force the paper's 2,552 or 47 counts. Obtain the paper's exact source snapshot/preprocessing or a clarification before claiming those figures have been reproduced.

The 1,453 decision sets are candidate records, not 1,453 independent or causal treatment effects. A later response may follow multiple treatment sets, incomplete timelines can hide intervening treatment, and response times vary. The output flags these issues. The strict complete-history subset of 2 is far too small to train or validate a treatment outcome model.

## Current implementation boundary

Implemented:

- Full field audit and paper-feature mapping.
- Paper-style 11-class treatment normalization with inspectable source mappings.
- Patient-level split and same-day state/action/response candidate table.
- Timing, action-mapping, and outcome-link quality flags.

Not implemented or validated:

- The paper's exact 2,552-event preprocessing and 7,395 training observations.
- Exact reproduction of the paper's reported Transformer results. The local five-member Transformer has been trained, but its test labels and per-feature scores are too limited for strong response claims.
- Adversarial outcome model, BCQ, DQN, Double DQN, NFQ, or CQL training.
- A validated dose/action space, reward utility beyond the paper's simple tumor-free label, toxicity features, calibrated uncertainty, or off-policy evaluation.
- Reliable causal treatment effects. The paper's adversarial representation does not remove unmeasured confounding.
- Online learning, clinician querying, or a clinical interface.

The earlier experimental LLM transition adapter has one full eligible held-out report at `data/processed/objectives_2_3/full_holdout_comparison.json`. It predicted only the majority `WT-With Tumor` category, achieved 73.3% accuracy on 30 valid rows (71.0% when its failed request counts wrong), and had 0% recall for tumor-free rows. The training-majority baseline reached 74.2% accuracy. This result is why the LLM was not retained as the twin. It does not resolve weak factual action/outcome linkage.

The trained Stage 1 model has a finite report at `data/processed/transformer_twin/transformer_training_report.json`. The environment smoke test is recorded in `data/processed/transformer_twin/rl_environment_smoke_v2.json`: 15 steps completed, but all 15 rewards were unavailable because tumor status was predicted missing. This confirms integration only. Do not proceed to treatment-outcome training or an RL policy until treatment timing and labeled outcomes are sufficiently complete and the target includes adequate class support.

## Reproduce the local preprocessing

```powershell
python scripts/paper_method_pipeline.py --input clinical.project-tcga-ov.2026-10-09.json
```

The pipeline uses only the Python standard library. It does not train models.
