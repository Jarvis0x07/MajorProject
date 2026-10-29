# Patient Digital Twins and an RL Doctor (`digital_twin_rl/`)

**Updated:** 2026-10-30
**Status:** Implemented end to end and validated against observed survival. All policy results come from a learned simulator built on retrospective observational data. This is a research prototype, not a clinical tool.

The system has two parts:

1. **A digital twin for every patient.** Given the patient's current situation and a candidate regimen, the twin predicts tumor response, time to progression, the chance that the next event is death, survival, and the risk of severe side effects, each with uncertainty.
2. **An RL agent ("the doctor")** that learns a treatment strategy by treating the twins. The twin is the environment, and each treatment line is one step.

```text
TCGA-OV JSON ─► treatment-line histories ─► outcome-model ensemble ─┐
                (data.py)                    (outcome_model.py)       ├─► patient twin ─► environment ─► DQN doctor ─► held-out evaluation
                               side-effect priors (toxicity.py) ─────┘   (twin.py)       (environment.py) (dqn.py)     (evaluate.py)
```

## How to run

From the repository root, with `requirements-digital-twin-rl.txt` installed:

```powershell
python -m digital_twin_rl all                                  # every stage (about 40 min on a laptop CPU)
python -m digital_twin_rl what-if --patient 098acb76 --line 2  # every option for one patient
python -m pytest tests -q
```

The stages can also be run one at a time: `data`, `train-twin`, `twins`, `train-agent`, `evaluate`. Outputs go to `data/processed/digital_twin_rl/`.

## 1. Data: treatment lines

Each eligible patient (`index_date = Diagnosis`, 587 patients) becomes an ordered list of **treatment lines**. A line groups the dated drug or radiation records that were actually given and started within 42 days of each other. The action is the line's regimen, expressed in the paper's 11 drug classes.

| | Count |
|---|---:|
| Patients / with at least one dated line | 587 / 508 |
| Treatment lines | 1,346 |
| Lines with a response label (tumor-free / with tumor) | 310 (82 / 228) |
| First events after a line (progression or death) / of which deaths | 1,149 / 230 |
| Deaths (overall survival events, per line) | 936 |
| Patients with undated *given* therapy | 94 (previously reported as 575) |

**Data fix (ADR-012).** 955 records have `treatment_or_therapy = "no"`, which means the therapy was *not* given. Of these, 931 are "Radiation Therapy, NOS". Earlier preprocessing counted them as radiation and as undated history, which inflated the "undated history" blocker. One patient recorded as dead without a death day is counted as a death at last contact.

**Pre-line state:** age, FIGO stage, grade, residual tumor after surgery, line number, months since diagnosis, disease setting (primary, recurrent, progressive or unknown), last tumor status, prior exposure to each drug class, platinum-free interval (sensitive vs resistant), and treatment-free interval. Missing values carry explicit flags. ECOG and Karnofsky scores are never recorded at line start in this export.

**Outcomes per line:**
- **Response:** the first known follow-up 30–548 days after the line starts and before the next line (tumor-free = 1).
- **First event:** the first of an explicit progression/recurrence day, the start of the next line (time to next treatment), or death, along with whether it was death. Otherwise the line is censored.
- **Overall survival:** from line start, censored at last contact.

## 2. The twin

**Outcome model.** A 10-member ensemble of multi-task MLPs; each member is trained on a different bootstrap of training patients. Each member predicts:
- the response probability;
- discrete-time hazards of the first event over 0–3, 3–6, 6–12, 12–24, 24–48 and 48+ months, using a censoring-aware likelihood;
- the probability that an event in each of those time bins is death, since early first events are mostly progression and late ones are often death;
- overall-survival hazards.

Disagreement between members gives the uncertainty estimate.

**Held-out test patients (59 patients, 132 lines):**

| Prediction | Twin | Baseline |
|---|---|---|
| Tumor response, AUROC (28 labelled lines, 5 tumor-free) | **0.86** | 0.83 (logistic regression) |
| Tumor response, Brier score | **0.089** | 0.157 (prevalence), 0.092 (logistic) |
| Overall survival, C-index (12-month risk) | **0.70** | — |
| 12-month mortality, predicted vs observed | 22.1% vs 22.4% | — |
| First event, C-index | 0.60 | — |
| 12-month first-event rate, predicted vs observed | 67.8% vs 67.5% | — |
| First event is death, AUROC | 0.73 | — |
| Share of first events that are deaths, predicted vs observed | 18.2% vs 18.9% | — |

On the validation split, response AUROC is 0.93, the overall-survival C-index is 0.77, the first-event C-index is 0.66, and death-cause AUROC is 0.76. Shuffling regimens across held-out lines worsens the likelihood: test negative log-likelihood rises by 0.25 per line. So the regimen carries predictive signal beyond the patient state. This is an association, not a causal effect.

**Side effects.** The export has no adverse-event fields. `toxicity_profiles.json` holds editable, ordinal class-level profiles summarised from drug labelling. For example, taxanes are linked to neuropathy, anthracyclines to cardiac and skin effects, and topoisomerase inhibitors to low blood counts. The model turns these into per-organ probabilities of a severe (grade ≥ 3) event. Risk rises with age ≥ 70, cumulative exposure, earlier severe events, and performance decline after severe side effects. **Every number in that file is an assumption that needs clinician review.**

**Personal twins.** A twin is the ensemble instantiated at one patient's real state. When the twin starts after some observed lines, that patient's own earlier response and progression results recalibrate it with shrunken offsets. Overall survival is not used for calibration, because it would reveal follow-up after the decision. `twins/patient_twins.jsonl` holds one card for each of the 587 patients. Each card has the starting state, observed history, and predicted outcomes for every regimen, including the one actually given.

**Between lines.** The outcome model predicts what happens *during* a line. The next line's recorded state follows frequencies learned from training lines and stored with the model:
- the disease setting, conditional on how soon the event came;
- the last tumor status, conditional on response;
- whether an end date exists, which affects the platinum-free interval;
- treatment duration.

As a result, simulated states are encoded the same way the data records them.

**Validation against reality (Objective 2).** Each patient's observed regimens are replayed in their twin (30 rollouts per held-out patient). Simulated survival from the first line is then compared with the Kaplan–Meier estimate of the same patients' observed survival. Each cell is simulated / observed.

| Split | 6 m | 12 m | 18 m | 24 m | 36 m | 48 m | 60 m |
|---|---|---|---|---|---|---|---|
| Train (408 patients) | 0.97 / 0.97 | 0.92 / 0.92 | 0.85 / 0.86 | 0.78 / 0.80 | 0.63 / 0.63 | 0.50 / 0.48 | 0.28 / 0.35 |
| Validation (53) | 0.97 / 1.00 | 0.91 / 0.96 | 0.85 / 0.89 | 0.78 / 0.78 | 0.65 / 0.63 | 0.53 / 0.41 | 0.31 / 0.24 |
| Test (47) | 0.97 / 0.96 | 0.90 / 0.91 | 0.83 / 0.91 | 0.76 / 0.83 | 0.62 / 0.73 | 0.48 / 0.49 | 0.25 / 0.36 |

The twins reproduce observed survival closely through 4 years. The 5-year residual is within about ±0.1 and changes sign between splits. An earlier version drifted badly (36 months: 0.46 simulated vs 0.73 observed). The fixes are recorded in ADR-015. See `evaluation/figures/simulation_fidelity.png`.

**Example: `what-if --patient 098acb76 --line 2`.** The patient is 74 years old, with stage III, grade 3 disease and 1–10 mm residual tumor. They are at first recurrence after carboplatin + paclitaxel.

| Regimen | Expected survival, months (5-year window) | Response probability | Any severe side effect | Clinician-model probability |
|---|---:|---:|---:|---:|
| platinum + taxanes | 37.7 | 0.10 | 0.74 | 0.35 |
| bevacizumab + platinum + taxanes | 37.5 | 0.09 | 0.82 | < 0.01 |
| antimetabolites + platinum + taxanes | 36.4 | 0.07 | 0.78 | 0.01 |
| platinum (single agent) | 34.8 | 0.06 | 0.54 | 0.11 |

The unconstrained DQN chooses bevacizumab + platinum + taxanes. DQN + BCQ and the clinician model choose platinum + taxanes, which is what the patient received. In reality the patient progressed after 6.1 months and lived 29.2 months from that line, so the twin was optimistic for this patient.

## 3. The environment and the RL doctor

| Component | Definition |
|---|---|
| Episode | Starts at a real patient's observed decision point. One step is one treatment line. Ends at death, long-term remission, the 5-year horizon, or after 8 lines (reached in 3–6% of episodes). |
| Actions | The 14 regimens given at least 10 times in training lines: platinum + taxanes (419), platinum (86), anthracyclines (82), topoisomerase inhibitors (74), taxanes (72), antimetabolites + platinum (43), antimetabolites (36), hormonal agents (28), antimetabolites + platinum + taxanes (22), alkylating agents (16), alkylating agents + platinum (14), radiation (12), topoisomerase inhibitors + vinca alkaloids (11), bevacizumab + platinum + taxanes (10). |
| Transition | The twin samples side effects, response, and the first event's time and cause (one ensemble member per step). If the event is progression, the state advances to the next line. |
| Reward per line | years alive + 0.25 × response − 0.15 × severe side effects − 0.5 × ensemble disagreement |
| Safety masks (illustrative) | No anthracycline after 2 prior anthracycline lines or a severe cardiac event. No taxane or vinca alkaloid after severe neuropathy. No platinum after severe hypersensitivity. Single agents only after repeated severe toxicity. Fallbacks are counted. |
| BCQ filter (`dqn_bcq` only) | Allows only regimens whose clinician-model probability is at least 0.3 × that of the most likely regimen in that state. It relaxes in about 0.1 steps per episode, when no supported regimen is safe. |

**Agents.** Both are masked Double DQN networks with two hidden layers of 128 units, trained for 30,000 episodes each. They train only on training-split patients, inside ensemble members 0–4. Training uses the twin's *expected* reward for each line, while the next state is still sampled, and the Q-learning targets are 8-step returns.

A first version, with sampled rewards and one-step targets, did not learn. It did no better than random and chose single agents over carboplatin + paclitaxel at first line, even though the twin rated the doublet best.

**Baselines:**
- `random`: picks uniformly among allowed regimens.
- `clinician`: the most likely regimen under a multinomial logistic model of what clinicians chose in that state.
- `guideline`: a simplified heuristic.
  - Platinum doublet at first line and for platinum-sensitive relapse.
  - An unused single non-platinum agent for platinum-resistant relapse.
  - Hormonal therapy after repeated severe toxicity.

## 4. Results on held-out patients

**Protocol.** The agents are tested on test-split patients they never saw: 47 starting at their first line and 28 starting at their first relapse. Each patient gets 30 rollouts. Random seeds are shared across doctors, so the comparisons are paired. 95% bootstrap confidence intervals are computed over patients. All scores come from the **evaluation twin** (ensemble members 5–9), which the agents never interacted with.

"Survival" below means simulated months alive within the 5-year window.

**From first line (47 patients):**

| Doctor | Survival (months) | Response | Severe side effects | Alive at 5 years | Choices clinicians would make |
|---|---:|---:|---:|---:|---:|
| random | 31.0 | 0.22 | 2.08 | 15% | 15% |
| clinician | 41.5 | 0.51 | 2.87 | 31% | 97% |
| guideline | 42.6 | 0.51 | 2.24 | 35% | 79% |
| **dqn** | **43.6** | 0.49 | 2.53 | 35% | 50% |
| **dqn_bcq** | **43.7** | 0.51 | 2.78 | 34% | 98% |

Paired differences against the clinician model:
- **dqn: +2.0 months** (95% CI 0.5 to 3.4), with 0.33 fewer severe side effects.
- **dqn_bcq: +2.2 months** (95% CI 1.4 to 3.0), with no clear difference in side effects (−0.09; CI −0.20 to 0.02).

**From first relapse (28 patients):**

| Doctor | Survival (months) | Response | Severe side effects | Alive at 5 years | Choices clinicians would make |
|---|---:|---:|---:|---:|---:|
| random | 30.4 | 0.04 | 2.27 | 14% | 25% |
| clinician | 32.9 | 0.07 | 2.81 | 21% | 97% |
| guideline | 33.9 | 0.10 | 2.26 | 24% | 80% |
| **dqn** | **36.8** | 0.09 | 2.54 | 26% | 32% |
| **dqn_bcq** | **36.9** | 0.11 | 2.74 | 27% | 98% |

Paired differences against the clinician model:
- **dqn: +4.0 months** (95% CI 1.5 to 6.7).
- **dqn_bcq: +4.0 months** (95% CI 2.2 to 5.9).

The guideline heuristic beats the clinician model mainly by causing fewer severe side effects.

**Exploitation check.** Inside the training twin (members 0–4), the gains are:
- dqn: +3.8 months at first line and +6.4 at relapse, against +2.0 and +4.0 in the evaluation twin.
- dqn_bcq: +2.7 and +5.1, against +2.2 and +4.0.

The unconstrained agent partly exploits the specific members it trained with. The BCQ agent's gains carry over better.

**What the agents do differently.**
- *dqn:* at first line it gives bevacizumab + platinum + taxanes 30% of the time. That regimen appears in only 10 training lines, so this choice extrapolates beyond the data.
- *dqn_bcq:* at first line it gives carboplatin + paclitaxel 100% of the time, like clinicians. At relapse it prefers platinum + taxanes (79%) over the single agents topotecan and liposomal doxorubicin, which the clinician model chose in 42% of these test relapse decisions.

**Main caveat: platinum for platinum-resistant relapse.** Share of all observed relapse decisions given a platinum-containing regimen:

| Who | Platinum-sensitive (≥ 6 months) | Platinum-resistant (< 6 months) |
|---|---:|---:|
| Clinicians (observed) | 47% | 16% |
| dqn | 55% | **68%** |
| dqn_bcq | 72% | 33% |

Platinum is not recommended for platinum-resistant disease. Because platinum was mostly given to patients with a better outlook, the twin associates platinum with better outcomes (confounding by indication). The unconstrained agent extrapolates that association to patients where clinicians rarely used platinum. The BCQ constraint halves the effect. **`dqn_bcq` is the recommended agent.**

Figures are in `data/processed/digital_twin_rl/evaluation/figures/`: simulation fidelity, the learning curve, policy comparisons, and first-regimen heatmaps. Per-patient recommendations are in `evaluation/test_patient_recommendations.csv`.

## 5. Limits

- **Observational data.** Regimen effects in the twin are associations. Unrecorded reasons for treatment choice remain (confounding by indication). Simulated gains are hypotheses for clinical review, not evidence of benefit.
- **The evaluation twin is not independent data.** Members 5–9 are trained on the same patients, with the same early stopping, as members 0–4. They guard against member-specific exploitation, not dataset bias.
- **Progression is a real-world proxy** (time to next treatment or recorded progression), not RECIST. Response labels exist for 23% of lines.
- **Side effects are assumed priors**, not data. Any agent behaviour that depends on them must be sensitivity-tested.
- **No dosing, cost or quality-of-life data.** Actions are regimen classes, not doses or schedules.
- **Small held-out sets:** 47 and 28 patients. The confidence intervals reflect simulation and patient sampling, not model misspecification.
- **Historical practice.** The export has a single PARP-inhibitor record (olaparib, mapped to `other`), and the first-line bevacizumab triplet appears in only 10 training lines. Modern maintenance strategies therefore cannot be learned.

## Files

| Path | Content |
|---|---|
| `digital_twin_rl/data.py` | Treatment lines, pre-line state, outcomes and event cause |
| `digital_twin_rl/outcome_model.py` | Ensemble, survival and competing-risk likelihood, between-line transition frequencies, held-out evaluation |
| `digital_twin_rl/toxicity.py`, `toxicity_profiles.json` | Side-effect priors |
| `digital_twin_rl/twin.py` | `PatientDigitalTwin`: `predict()`, `step()`, personal calibration |
| `digital_twin_rl/environment.py` | Gym-style environment, reward, safety and BCQ masks with fallback flags |
| `digital_twin_rl/policies.py` | Clinician behaviour model, random and guideline doctors |
| `digital_twin_rl/dqn.py` | Masked Double DQN with n-step targets |
| `digital_twin_rl/evaluate.py`, `registry.py` | Held-out evaluation, simulation fidelity, figures, twin cards, what-if |
| `data/processed/digital_twin_rl/` | Histories, model checkpoints, twin cards, agents, evaluation |
