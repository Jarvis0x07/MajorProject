# Patient record format (`ovarian-twin-record/v1`)

One JSON file describes one ovarian-cancer patient **up to today**. All times are **months since diagnosis**. "Today" is the moment a new treatment is about to be chosen. The website builds the patient's digital twin at that moment and predicts what each medication would do.

Start from `TEMPLATE.json`. The `patient_*.json` files are real patients from the test split, which the twin never trained on, cut off at a real decision point. `ANSWER_KEY.md` says what really happened next. It is for the presenter only; the website never reads it.

## Fields

| Field | Required | Allowed values |
|---|---|---|
| `format` | no | `"ovarian-twin-record/v1"` |
| `patient_label` | no | Any text shown on screen, e.g. `"Demo patient A"` |
| `age_at_diagnosis` | **yes** | 18–100 |
| `figo_stage` | **yes** | `I, IA, IB, IC, II, IIA, IIB, IIC, III, IIIA, IIIB, IIIC, IV` |
| `tumor_grade` | no | `G1, G2, G3, G4` |
| `residual_tumor_after_surgery` | no | `none`, `1-10 mm`, `11-20 mm`, `>20 mm` |
| `treatments` | no | List of past treatment courses (see below); empty or missing for a newly diagnosed patient |
| `follow_ups` | no | List of check-ups (see below) |
| `today_month` | **yes** | Months since diagnosis now; must not be earlier than any treatment or check-up |
| `current_situation` | no | `newly diagnosed`, `recurrence`, `progression`, `unknown`. If missing, it is worked out from the last check-up event. |

**`treatments[]`**, one entry per course:

| Field | Required | Meaning |
|---|---|---|
| `drugs` | **yes** | Drug names given together, e.g. `["Carboplatin", "Paclitaxel"]`. Use `"Radiation"` for radiotherapy. |
| `start_month` | **yes** | When the course started |
| `end_month` | no | When it ended |
| `reason` | no | `first treatment`, `recurrence`, `progression` |

Courses starting within about 6 weeks of each other count as one treatment line, as in the dataset.

**`follow_ups[]`**, one entry per check-up:

| Field | Required | Meaning |
|---|---|---|
| `month` | **yes** | When the check-up happened |
| `tumor_status` | no | `tumor-free` or `tumor present` |
| `event` | no | `recurrence` or `progression` (the cancer came back or grew) |

## Drug names the twin knows

Altretamine, Aminocamptothecin, Aminopterin, Anastrozole, Bevacizumab, Capecitabine, Carboplatin, Chlorambucil, Cisplatin, Cyclophosphamide, Docetaxel, Doxorubicin, Doxorubicin Hydrochloride, Etoposide, Exemestane, Fluorouracil, Fulvestrant, Gemcitabine, Gemcitabine Hydrochloride, Ifosfamide, Irinotecan Hydrochloride, Irofulven, Letrozole, Leucovorin, Leuprolide Acetate, Medroxyprogesterone Acetate, Megestrol Acetate, Melphalan, Motesanib Diphosphate, Nab-Paclitaxel, Oxaliplatin, Paclitaxel, Paclitaxel Poliglumex, Pan-Vegfr/Tie2 Tyrosine Kinase Inhibitor Cep-11981, Pegylated Liposomal Doxorubicin Hydrochloride, Pemetrexed, Pemetrexed Disodium, Plevitrexed, Radiation, Sorafenib, Tamoxifen, Taxane Compound, Topotecan, Topotecan Hydrochloride, Trabectedin, Vincristine, Vinorelbine, Vinorelbine Tartrate, Vosaroxin.

Names are matched without regard to upper or lower case. An unknown name is rejected with a message listing these names.

## What the twin cannot use

Doses, blood tests (e.g. CA-125), scans, genetics and side-effect history are not in the training data, so the format has no fields for them.
