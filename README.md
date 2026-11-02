# Digital Twins and Reinforcement Learning for Ovarian Cancer Treatment

A two-semester Major Project. It builds a **digital twin** for each ovarian-cancer patient: a computer model of that patient that predicts what would happen under a given treatment. It then trains a **reinforcement-learning (RL) agent** that learns treatment strategies by "treating" the twins instead of real patients.

```text
patient records ──► digital twin per patient ──► twin used as a simulator ──► RL agent learns by treating twins
   (TCGA-OV)          (Objective 1)                 (Objective 2)                 (Objective 3)
```

> Research prototype on public, de-identified data. Not medical advice.

---

## Project status (end of semester 1)

| Objective | Status | What exists |
|---|---|---|
| 1. Patient digital twin | **Done** | A twin for every patient, plus on-the-spot twins for uploaded records, tested on unseen patients |
| 2. Twin as a simulation environment | **Done** | Simulates treatment line after treatment line; simulated survival matches real survival |
| 3. RL framework | **Done** | States, actions, reward, safety and guideline constraints, baselines; basic DQN agent trained and checked |
| 4. Safe RL treatment optimisation | Next semester | Early version: an agent restricted to treatments doctors actually use (BCQ) |
| 5. Full evaluation | Next semester | Early version: held-out comparison against baselines |

A **demo website** shows Objectives 1–3 live: upload a patient record, get a twin, test medications on it.

---

## What has been done, step by step

| When | Step |
|---|---|
| Sep 2026 | Project plan and objectives |
| Oct 2 | Literature survey: 11 papers, including the base paper (Qin et al. 2026, arXiv:2606.17405) |
| Oct 18 | Chose the dataset (TCGA-OV, 608 ovarian-cancer patients); first data cleaning and patient description |
| Oct 24–27 | Data audits (fields, follow-up timing, treatment timing); simple forecasting baselines. An AI language model (MedGemma) was tried as the twin and dropped: it did no better than guessing |
| Oct 27–29 | Followed the base paper: 11 drug classes and a Transformer twin. It trained, but could not give usable outcome signals |
| Oct 30 | **Final twin** (`digital_twin_rl/`). Found and fixed a data bug: 931 "radiation **not** given" records had been counted as treatments. Built the outcome model, per-patient twins, a simulator checked against real survival, and the RL framework with agents and baselines. An independent code review led to fixes |
| Nov 3 | **Mid-semester demo** (`demo/`): on-the-spot twins built from the 5 most similar stored twins, a guideline constraint layer, and demo patient files |

Reasons for each design decision are recorded in [`docs/architecture/DECISION_LOG.md`](docs/architecture/DECISION_LOG.md) (ADR-001 to ADR-016).

---

## Key results

All results below are on **patients the models never trained on**.

**Objective 1: the digital twin is accurate.**

| Prediction | Result |
|---|---|
| Will the tumour disappear? (AUROC; 0.5 = coin flip, 1 = perfect) | **0.86** |
| Who dies sooner? (survival C-index) | **0.70** |
| Death within 1 year: predicted vs actual | **22.1% vs 22.4%** |

**Objective 2: the simulator matches reality.** Each training patient's real treatments were replayed in their twin, and the share still alive was compared with what actually happened:

| Months after first treatment | 12 | 24 | 36 | 48 | 60 |
|---|---|---|---|---|---|
| Twins (simulated) | 92% | 78% | 63% | 50% | 28% |
| Real patients | 92% | 80% | 63% | 48% | 35% |

**Objective 3: the RL framework works.** The agent's reward rises during training. On 47 unseen patients it beats the baselines, measured as simulated months alive within 5 years from first treatment:

| Doctor | Months |
|---|---|
| Random treatment | 31.0 |
| Copying usual practice | 41.5 |
| Simplified guideline | 42.6 |
| **Basic RL agent (DQN)** | **43.5** |

Full details are in [`docs/architecture/DIGITAL_TWIN_RL.md`](docs/architecture/DIGITAL_TWIN_RL.md).

---

## How to run

> **Run every command from this folder**, the one that contains this `README.md`. In VS Code, the terminal may open one level up; if so, `cd MajorProject` first. Otherwise you get `No module named demo`.

### 1. Install (once)

Python 3.12 or newer:

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the demo website

```powershell
python -m demo
```

Open **http://localhost:8000**, then:

1. Upload a patient file from [`demo/sample_patients/`](demo/sample_patients/), for example `patient_C_late_relapse.json`.
2. Watch the new twin being built from the 5 most similar stored twins.
3. Pick a medication to see the twin's predictions: response, time to progression, survival, and side-effect risks.
4. Try **Simulate next 5 years**, **Compare all medications** and **Ask the AI doctor**.
5. The **How it works** tab shows the evidence for Objectives 1–3.

Press `Ctrl+C` in the terminal to stop. If it says the address is already in use, another copy is running: close it, or use `python -m demo --port 8001`.

**Demo files:**
- Five real test patients (newly diagnosed, older patient with advanced disease, late relapse, early relapse, heavily pre-treated).
- [`TEMPLATE.json`](demo/sample_patients/TEMPLATE.json) and [`FORMAT.md`](demo/sample_patients/FORMAT.md), to write your own.
- [`ANSWER_KEY.md`](demo/sample_patients/ANSWER_KEY.md), which says what really happened to each patient. It is for the presenter only.

A suggested 5-minute walkthrough is in [`demo/README.md`](demo/README.md).

### 3. Ask a twin from the command line

```powershell
python -m digital_twin_rl what-if --patient 098acb76 --line 2
```

This prints the twin's predictions for all 14 medications for that patient at that treatment line.

### 4. Rebuild everything from the raw data (optional, about 40 minutes)

Trained models are already included, so this is only needed after changing the code.

```powershell
python -m digital_twin_rl all
```

| Step (run on its own as `python -m digital_twin_rl <step>`) | What it does |
|---|---|
| `data` | Turns the raw JSON into treatment-line histories |
| `train-twin` | Trains the outcome model and tests it on unseen patients |
| `twins` | Creates a twin card for each of the 587 patients |
| `twin-db` | Builds the stored-twin database used by the demo |
| `train-agent` | Trains the RL agents inside the twins |
| `evaluate` | Compares agents with baselines, checks simulator fidelity, draws figures |

To regenerate the demo patient files: `python -m demo.make_sample_patients`.

### 5. Run the tests

```powershell
python -m pytest tests -q
```

There are 29 tests: data processing, survival maths, twin and environment, RL agent, patient-file format, and the demo API.

### 6. Change the website (optional)

The website is already built into `demo/web/dist`, so Node is only needed to edit it. Run these in `demo/web` (Node 22 or newer, and pnpm): `pnpm install`, `pnpm dev` (live preview on http://localhost:5173, with `python -m demo` running), and `pnpm build`.

---

## Folder structure

```text
├── data/
│   ├── raw/                  TCGA-OV clinical export (608 patients), the only input
│   └── processed/            Generated outputs: twin model, twins, agents, evaluation (see its README)
├── digital_twin_rl/          Main code: data → outcome model → twins → simulator → RL agents → evaluation
├── demo/                     Mid-semester demo: Python backend, website (web/), sample patient files
├── scripts/                  Earlier stages of the project (kept as history, see below)
├── tests/                    Automated tests
└── docs/
    ├── architecture/         Design documents and the decision log
    ├── data/                 Dataset audits, patient-state schema, dataset notes
    ├── literature/           The 11 papers from the literature survey
    └── university_forms/     Major-project guidelines and forms
```

### Earlier stages (`scripts/`)

These document how the project got here. They are not needed to run the demo.

| Script | What it was for |
|---|---|
| `preprocess_tcga_ov.py`, `patient_twin.py` | First data extraction and patient summaries |
| `audit_tcga_ov_fields.py`, `audit_followup_timeline.py`, `audit_treatment_transitions.py` | Data-quality audits |
| `evaluate_trajectory_baseline.py`, `compare_llm_trajectory.py`, `llm_twin_rl.py` | Simple baselines and the LLM-twin experiment (not selected) |
| `paper_method_pipeline.py` | Base-paper preprocessing; its drug-class mapping is reused by the final twin |
| `transformer_digital_twin.py`, `transformer_twin_rl_environment.py` | The base paper's Transformer twin and its environment test |

The data scripts read `data/raw/clinical.project-tcga-ov.2026-10-09.json`; the Transformer scripts read the table made by `paper_method_pipeline.py`. Exact commands are in [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md).

---

## The data

TCGA-OV clinical records for 608 women with ovarian cancer, diagnosed 1992–2013. The dataset is public and de-identified.

- **Used:** age, stage, grade, tumour left after surgery, 3,170 treatments given (69 drugs), 2,036 check-ups (tumour status, recurrence), and survival.
- **Not in the data:** side effects, doses in comparable units, blood tests such as CA-125, scans, cost and quality of life.

## Limitations

- **Associations, not proof.** The twin learns from historical records, so a link between a drug and good outcomes may reflect who received that drug. For example, platinum was mostly given to patients with a better outlook. The guideline constraint layer flags known cases.
- **Side-effect risks are assumptions** from drug-class profiles, because the dataset records no side effects.
- **All RL results come from simulation.** They suggest strategies to study; they are not evidence that a treatment works better.

## Further reading

| Document | Contents |
|---|---|
| [`docs/architecture/DIGITAL_TWIN_RL.md`](docs/architecture/DIGITAL_TWIN_RL.md) | Final twin and RL system: method, results, limits |
| [`demo/README.md`](demo/README.md) | Demo website: how it works, walkthrough, API |
| [`docs/architecture/DECISION_LOG.md`](docs/architecture/DECISION_LOG.md) | Every major decision and why it was made |
| [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) | Every script and output, with commands |
| [`docs/architecture/PAPER_METHOD_REPLICATION.md`](docs/architecture/PAPER_METHOD_REPLICATION.md) | How the base paper's method was adapted |
| [`docs/data/patient_state_schema.md`](docs/data/patient_state_schema.md) | How a patient's situation is encoded |
