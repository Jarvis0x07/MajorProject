# Answer key (presenter only)

What really happened to each demo patient after the decision point in the file. The website never reads this file; the patients are from the test split, which the twin never trained on.

| File | Scenario | Regimen actually given | Response | Months to progression/death | Months survived from this decision |
|---|---|---|---|---|---|
| patient_A_newly_diagnosed.json | Newly diagnosed, stage III, small residual tumour | platinum+taxanes | not recorded | 6.1 (event) | 57.6 (died) |
| patient_B_advanced_older.json | Newly diagnosed, older patient with advanced disease | other+platinum+taxanes | tumour-free | 7.2 (censored) | 7.2 (alive at last contact) |
| patient_C_late_relapse.json | Relapse more than a year after platinum (platinum-sensitive) | platinum+taxanes | not recorded | 77.6 (censored) | 77.6 (alive at last contact) |
| patient_D_early_relapse.json | Progression within 6 months of platinum (platinum-resistant) | topoisomerase_inhibitors | not recorded | 4.8 (event) | 9.6 (died) |
| patient_E_heavily_pretreated.json | Fourth or later treatment line | antimetabolites+platinum | not recorded | 3.6 (event) | 10.6 (died) |
