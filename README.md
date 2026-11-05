# Digital Twins and Reinforcement Learning for Ovarian Cancer Treatment

Builds a **digital twin** for every ovarian-cancer patient in the TCGA-OV dataset: a model of that patient that predicts what each treatment would do. The twins are then used as the **environment for a reinforcement-learning (RL) agent**, which learns treatment strategies by treating twins instead of real patients. A **demo website** takes an uploaded patient record, builds that patient's twin and ranks how effective each medication would be.

```text
TCGA-OV records ──► outcome model ──► one twin per patient ──► RL agent learns   ──► website: upload a patient,
 (608 patients)     (shared)          (from 5 most similar      by treating twins      see each medication's
                                       training patients)                              predicted effectiveness
```

> Research prototype on public, de-identified data. Not medical advice.

## How it works

| Step | File | What it does |
|---|---|---|
| 1. Data | `digital_twin_rl/data.py` | Groups each patient's drug records into **treatment lines**. For every line it keeps the state before it (age, stage, grade, residual tumour, line number, situation, past drugs, months since platinum), the regimen given, and what happened next (tumour response, progression-free survival, overall survival). |
| 2. Outcome model | `digital_twin_rl/model.py` | An ensemble of 10 small neural networks predicts, for a state + regimen: chance the tumour disappears, risk of progression or death over time, and survival. Patients still alive at last contact are handled with a discrete-time survival likelihood. |
| 3. Digital twins | `digital_twin_rl/twin.py` | Each twin is the outcome model at that patient's own state, **adjusted using the 5 most similar training patients**. For those 5, the progression and death events that really happened are compared with the number the model expected for them. If they did worse than expected the twin's risk goes up, and if better it goes down: `multiplier = (observed + 20) / (expected + 20)`. A twin can predict a treatment's effect or simulate a treatment line and move on to the next decision. |
| 4. RL agent | `digital_twin_rl/rl.py` | Each episode is one patient's twin. The agent picks a regimen; the twin simulates response, then progression or death, and moves to the next decision, until death or 5 years. Reward = years alive + a bonus for tumour response − a penalty where the model is unsure. Platinum is not allowed when the cancer returned within 6 months of platinum (platinum-resistant), as guidelines advise. The agent is a deep Q-network (DQN). |
| Website | `demo/` | Upload a patient record, and a twin is built from the 5 most similar training patients. The site shows every medication's predicted effectiveness and the RL agent's choice. |

## Results

All results are on patients the models never trained on (data split 80% train / 10% validation / 10% test).

**The outcome model is accurate** (132 treatment lines of 59 test patients):

| Prediction | Result |
|---|---|
| Will the tumour disappear? (AUROC; 0.5 = coin flip, 1 = perfect) | **0.89** |
| Who dies sooner? (survival C-index) | **0.69** |
| Death within 1 year: predicted vs actual | **21.6% vs 22.4%** |

**Building twins from 5 similar patients keeps them as accurate as the shared model** (Brier score, lower is better). Five patients carry little information, so the adjustment is deliberately cautious:

| 12-month prediction | Shared model only | Twin from 5 similar patients |
|---|---|---|
| Progression | 0.213 | 0.213 |
| Death | 0.135 | 0.135 |

**The twins are a realistic environment.** Each held-out patient's real treatments were replayed in their twin (100 patients), and the share still alive was compared with reality:

| Months after first treatment | 12 | 24 | 36 | 48 | 60 |
|---|---|---|---|---|---|
| Real patients | 93% | 81% | 68% | 45% | 30% |
| Twins (simulated) | 92% | 81% | 69% | 58% | 37% |

The twins match reality closely for 3 years and are optimistic after that, when few patients are left to compare.

**The RL agent learns a better strategy inside the twins.** Average months alive within 5 years when each doctor treats the twins of the 59 test patients (20 simulated runs each):

| Doctor | Months alive |
|---|---|
| Random treatment | 31.9 |
| Standard care (carboplatin + paclitaxel; liposomal doxorubicin if platinum-resistant) | 45.1 |
| **DQN agent** | **49.5** |

The agent gives carboplatin + paclitaxel first in 97% of cases, as guidelines do. Its gain comes mainly from relapse decisions, e.g. paclitaxel instead of liposomal doxorubicin for platinum-resistant relapse.

## How to run

Run every command from this folder (the one with this README). Python 3.12 or newer.

```powershell
python -m pip install -r requirements.txt
```

### Demo website

```powershell
python -m demo
```

Open **http://localhost:8000** and upload a file from [`demo/sample_patients/`](demo/sample_patients/), e.g. `patient_C_late_relapse.json`. You get the patient's twin, the 5 similar patients it was built from, and every medication ranked by predicted effectiveness. To write your own patient file, start from `TEMPLATE.json`; the fields are explained in `FORMAT.md`. `ANSWER_KEY.md` says what really happened to each sample patient (for the presenter). Use `--port 8001` if port 8000 is busy.

### Rebuild everything from the raw data (optional)

The trained files are already in `data/processed/`, so this is only needed after changing the code. It takes about 15 minutes.

```powershell
python -m digital_twin_rl all
```

Or one step at a time: `prepare`, `train-model`, `build-twins`, `train-agent`. Results are written to `data/processed/results.json`.

### Tests

```powershell
python -m pytest tests -q
```

### Change the website (optional)

The website is pre-built in `demo/web/dist`. To edit it you need Node 22+ and pnpm: run `pnpm install` once in `demo/web`, then `pnpm dev` (live preview at http://localhost:5173 while `python -m demo` runs) and `pnpm build`.

## Folder structure

```text
├── data/
│   ├── raw/                TCGA-OV clinical export (608 patients), the only input
│   └── processed/          histories, outcome model, twins, RL agent, results.json
├── digital_twin_rl/        data.py → model.py → twin.py → rl.py  (+ record.py for uploads)
├── demo/
│   ├── server.py           website backend (one endpoint: POST /api/analyse)
│   ├── web/                website (React); dist/ is the built version
│   └── sample_patients/    demo patient files, format and template
├── tests/
└── docs/                   dataset notes, literature, university forms
```

## Limitations

- **Associations, not proof.** The model learns from historical records, so a drug linked with good outcomes may simply have been given to patients with a better outlook (for example, platinum). The platinum-resistance rule guards against the best-known case.
- **Five similar patients carry little information.** The neighbour adjustment is deliberately cautious, and on test patients it changes accuracy very little (see the results).
- **All RL results come from simulation**, inside the same model the agent trained in. They suggest strategies to study, not evidence that a treatment works better.
- **Not in the data:** doses, blood tests (CA-125), scans, genetics, side effects and performance status.
