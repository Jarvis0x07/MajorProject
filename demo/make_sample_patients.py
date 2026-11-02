"""Write the demo patient files: real test-split patients the twin never trained on, cut off at a decision point.

Run from the repository root:  python -m demo.make_sample_patients

Writes demo/sample_patients/patient_*.json (the files to upload), TEMPLATE.json,
and ANSWER_KEY.md (what actually happened next; for the presenter only, never
read by the website). Each file is checked to reproduce the dataset's own state
for that decision point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from digital_twin_rl.data import load_histories
from digital_twin_rl.outcome_model import encode_state
from digital_twin_rl.record import FORMAT, RecordError, case_to_record, parse_record

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "sample_patients"

SCENARIOS = [
    ("patient_A_newly_diagnosed", "Newly diagnosed, stage III, small residual tumour",
     lambda s: s["line_number"] == 1 and s["setting"] == "primary" and s["stage"] == 3
     and s["residual"] in (0, 1) and 45 <= (s["age_years"] or 0) <= 65),
    ("patient_B_advanced_older", "Newly diagnosed, older patient with advanced disease",
     lambda s: s["line_number"] == 1 and s["setting"] == "primary" and (s["age_years"] or 0) >= 68
     and (s["stage"] == 4 or (s["residual"] or 0) >= 2)),
    ("patient_C_late_relapse", "Relapse more than a year after platinum (platinum-sensitive)",
     lambda s: s["setting"] == "recurrent" and s["line_number"] == 2
     and (s["months_since_last_platinum"] or 0) >= 12),
    ("patient_D_early_relapse", "Progression within 6 months of platinum (platinum-resistant)",
     lambda s: s["setting"] == "progressive" and s["months_since_last_platinum"] is not None
     and s["months_since_last_platinum"] < 6),
    ("patient_E_heavily_pretreated", "Fourth or later treatment line",
     lambda s: s["line_number"] >= 4),
]

TEMPLATE = {
    "format": FORMAT,
    "patient_label": "Your patient name or code",
    "age_at_diagnosis": 60,
    "figo_stage": "IIIC",
    "tumor_grade": "G3",
    "residual_tumor_after_surgery": "1-10 mm",
    "treatments": [
        {"drugs": ["Carboplatin", "Paclitaxel"], "start_month": 1.0, "end_month": 5.5, "reason": "first treatment"}
    ],
    "follow_ups": [
        {"month": 7.0, "tumor_status": "tumor-free"},
        {"month": 20.0, "tumor_status": "tumor present", "event": "recurrence"}
    ],
    "today_month": 21.0,
    "current_situation": "recurrence",
}


def main() -> None:
    dataset = ROOT / "data" / "raw" / "clinical.project-tcga-ov.2026-10-09.json"
    cases = {str(c.get("case_id")): c for c in json.loads(dataset.read_text(encoding="utf-8"))}
    histories = load_histories(ROOT / "data" / "processed" / "digital_twin_rl" / "patient_histories.jsonl")
    test = sorted((p for p in histories if p["split"] == "test" and p["lines"]), key=lambda p: p["patient_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    used, key_rows = set(), []
    for filename, description, matches in SCENARIOS:
        for patient in test:
            if patient["patient_id"] in used:
                continue
            line = next((l for l in patient["lines"] if matches(l["state"])), None)
            if line is None:
                continue
            letter = filename.split("_")[1]
            try:
                record = case_to_record(cases[patient["patient_id"]], line["start_day"], f"Demo patient {letter}",
                                        line["state"]["setting"])
            except RecordError:
                continue
            parsed = parse_record(record)
            gap = float(np.abs(np.array(encode_state(parsed["state"])) - np.array(encode_state(line["state"]))).max())
            if gap > 0.01:      # the file must reproduce the dataset's own state for this decision point
                continue
            used.add(patient["patient_id"])
            (OUT / f"{filename}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            o = line["outcome"]
            key_rows.append(f"| {filename}.json | {description} | {'+'.join(line['regimen'])} | "
                            f"{'tumour-free' if o['response'] == 1 else 'tumour present' if o['response'] == 0 else 'not recorded'} | "
                            f"{o['pfs_months']:.1f} ({'event' if o['pfs_event'] else 'censored'}) | "
                            f"{o['os_months']:.1f} ({'died' if o['os_event'] else 'alive at last contact'}) |")
            print(f"{filename}.json  <- test patient {patient['patient_id'][:8]}, line {line['line_index'] + 1}, "
                  f"state reproduced (max feature gap {gap:.4f})")
            break
        else:
            print(f"No test patient matched scenario {filename}")
    (OUT / "TEMPLATE.json").write_text(json.dumps(TEMPLATE, indent=2) + "\n", encoding="utf-8")
    (OUT / "ANSWER_KEY.md").write_text(
        "# Answer key (presenter only)\n\nWhat really happened to each demo patient after the decision point in the "
        "file. The website never reads this file; the patients are from the test split, which the twin never "
        "trained on.\n\n| File | Scenario | Regimen actually given | Response | Months to progression/death | "
        "Months survived from this decision |\n|---|---|---|---|---|---|\n" + "\n".join(key_rows) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
