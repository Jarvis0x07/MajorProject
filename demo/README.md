# Ovarian Twin Lab: mid-semester demo (Objectives 1–3)

Upload a patient record and the site builds that patient's **digital twin on the spot** from the 5 most similar of 1,070 stored twins. You then choose a medication and the twin predicts:

- the chance the tumour disappears;
- months until the cancer returns or grows;
- survival;
- severe side-effect risks.

It can also play the next five years forward and show what the RL agent would choose.

## Run it

From the repository root, with Python 3.12 or newer (tested on 3.14). The website is pre-built in `web/dist`:

```powershell
python -m pip install -r requirements.txt
python -m demo
```

Open http://localhost:8000. Everything runs on this computer; uploaded files go only to the local server.

To change the website, run `pnpm install` once in `demo/web`. Then use `pnpm dev` (live reload on http://localhost:5173, with `python -m demo` running alongside) and `pnpm build` to refresh `web/dist`.

## Demo files

`sample_patients/` holds real patients from the **test split**, which the twin never trained on, each cut off at a real decision point. Every file reproduces the dataset's own patient state exactly.

| File | Situation |
|---|---|
| `patient_A_newly_diagnosed.json` | Newly diagnosed, stage III, small residual tumour |
| `patient_B_advanced_older.json` | Newly diagnosed, older patient, advanced disease |
| `patient_C_late_relapse.json` | Relapse more than a year after platinum (platinum-sensitive) |
| `patient_D_early_relapse.json` | Progression within 6 months of platinum (platinum-resistant), 5th line |
| `patient_E_heavily_pretreated.json` | 4th treatment line |
| `TEMPLATE.json` | Blank starting point for your own files |

`FORMAT.md` documents every field. `ANSWER_KEY.md` says what really happened to each demo patient. It is for the presenter only; the website never reads it.

To regenerate the demo files: `python -m demo.make_sample_patients`.

## Suggested walkthrough (about 5 minutes)

1. **How it works** tab: the three objectives with real numbers. These are twin accuracy on unseen patients, simulated vs observed survival, and the RL learning curve and baselines.
2. **Twin Lab:** upload `patient_C_late_relapse.json`. The map shows all 1,070 stored twins; the 5 nearest light up and the new twin forms between them. The patient card shows what the twin knows about today.
3. Choose **Carboplatin + Paclitaxel**, then compare with **Liposomal Doxorubicin**. The predictions come from this patient's twin only.
4. **Simulate next 5 years:** 300 simulated futures, plus one future traced line by line (Objective 2).
5. Upload `patient_D_early_relapse.json`. Platinum options carry a **guideline caution** (a treatment constraint, Objective 3), and **Ask the AI doctor** avoids them.

## How a new twin is built

1. The record goes through the same code as the training data (`digital_twin_rl/record.py` → `data.build_patient` / `state_at`). This gives today's 37-feature state.
2. The 5 nearest stored decision points come from distinct training patients, by z-scored distance.
3. The shared outcome ensemble is evaluated at this patient's own state. Its progression risk is calibrated with the observed vs expected events of those 5 twins. The expected events use out-of-bag predictions, i.e. only from ensemble members that never trained on them. Response and survival are not adjusted, because on validation patients 5 neighbours made those worse.
4. All results come from this one twin. Neighbours' outcomes are never shown.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/info` | Project numbers, medications, twin map |
| POST | `/api/twins` | Upload a record (multipart `file`) → new twin |
| POST | `/api/twins/{id}/predict` | `{"medication": key}` → prediction |
| GET | `/api/twins/{id}/compare` | All 14 medications, ranked |
| POST | `/api/twins/{id}/simulate` | `{"medication": key, "runs": 300}` → 5-year simulation |
| GET | `/api/twins/{id}/ai-doctor` | Basic DQN agent's choice under the constraints |

Research prototype on public, de-identified TCGA-OV data. Not medical advice.
