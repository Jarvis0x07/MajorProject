#!/usr/bin/env python3
"""Show the patient-twin state and forecast interface on fictional data.

This self-contained demo requires no dataset, API key, or third-party package.
The forecast is a fixed mock response to demonstrate formatting only.
"""
import json

from patient_twin import build_patient_state, predict_next_trajectory


def mock_model(_prompt: str) -> str:
    return json.dumps({
        "prediction": None,
        "confidence": None,
        "evidence_fields": ["figo_stage", "therapeutic_agents"],
        "limitations": ["Demo uses a mock model; no real prediction was made."],
    })


def main() -> None:
    fictional_case = {
        "case_id": "DEMO-ONLY-001",
        "primary_site": "Ovary",
        "demographic": {"age_at_index": 58, "sex_at_birth": "female", "race": "Not Reported"},
        "diagnoses": [{
            "classification_of_tumor": "Primary",
            "primary_diagnosis": "Serous cystadenocarcinoma, NOS",
            "figo_stage": "Stage IIIC",
            "tumor_grade": "G3",
            "days_to_diagnosis": 0,
            "treatments": [{
                "treatment_type": "Chemotherapy",
                "therapeutic_agents": "carboplatin; paclitaxel",
                "treatment_dose": 300,
                "treatment_dose_units": "mg",
                "number_of_cycles": 6,
                "days_to_treatment_start": 30,
            }],
        }],
        "follow_ups": [{
            "timepoint_category": "Post Initial Treatment",
            "disease_response": "TF-Tumor Free",
            "days_to_follow_up": 240,
        }],
    }

    state = build_patient_state(fictional_case)
    forecast = predict_next_trajectory(state, mock_model)
    print("DIGITAL TWIN DEMO - FICTIONAL DATA ONLY\n")
    print("1) Structured patient state created from the input record:\n")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print("\n2) Forecast interface result (mock; not a real model prediction):\n")
    print(json.dumps({
        "prediction": forecast.prediction,
        "confidence": forecast.confidence,
        "evidence_fields": forecast.evidence_fields,
        "limitations": forecast.limitations,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
