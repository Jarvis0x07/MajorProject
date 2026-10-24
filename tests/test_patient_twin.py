import json
import unittest

from scripts.patient_twin import build_patient_state, predict_next_trajectory


CASE = {
    "case_id": "case-1", "primary_site": "Ovary",
    "demographic": {"age_at_index": 60, "vital_status": "Dead"},
    "diagnoses": [{"figo_stage": "Stage IIIC", "treatments": [{
        "therapeutic_agents": "carboplatin", "treatment_dose": 300,
        "treatment_dose_units": "mg", "days_to_treatment_start": 12,
        "treatment_outcome": "Complete Response"}]}],
    "follow_ups": [{"disease_response": "WT-With Tumor", "days_to_follow_up": 90}],
}


class PatientTwinTests(unittest.TestCase):
    def test_state_preserves_missingness_units_and_separates_retrospective_outcomes(self):
        state = build_patient_state(CASE)
        self.assertEqual(state["profile"]["age_at_index"], {"value": 60, "status": "observed"})
        self.assertEqual(state["profile"]["race"]["status"], "absent")
        self.assertEqual(state["treatment_history"][0]["dose_units"]["value"], "mg")
        self.assertEqual(state["retrospective_outcomes_not_safe_as_current_state"][0]["response"]["value"], "WT-With Tumor")
        self.assertEqual(state["time_axis_status"], "unverified_relative_day_origins")

    def test_as_of_filtering_fails_closed_without_verified_time_origin(self):
        with self.assertRaises(ValueError):
            build_patient_state(CASE, as_of=30)

    def test_trajectory_prediction_validates_llm_json_and_adds_limitation(self):
        def fake_model(prompt):
            self.assertIn("Do not recommend treatment", prompt)
            return json.dumps({"prediction": "WT-With Tumor", "confidence": 0.4,
                               "evidence_fields": ["disease_response"], "limitations": []})
        result = predict_next_trajectory(build_patient_state(CASE), fake_model)
        self.assertEqual(result.prediction, "WT-With Tumor")
        self.assertEqual(result.confidence, 0.4)
        self.assertTrue(any("unverified" in x for x in result.limitations))

    def test_trajectory_prediction_rejects_invalid_confidence(self):
        def bad_model(_):
            return '{"prediction":"x","confidence":1.2,"evidence_fields":[],"limitations":[]}'
        with self.assertRaises(ValueError):
            predict_next_trajectory(build_patient_state(CASE), bad_model)


if __name__ == "__main__":
    unittest.main()
