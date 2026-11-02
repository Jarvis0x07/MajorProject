import copy
import json
import unittest
from pathlib import Path

from digital_twin_rl.outcome_model import STATE_FEATURES, encode_state
from digital_twin_rl.record import RecordError, parse_record

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "demo" / "sample_patients"
ARTIFACTS = ROOT / "data" / "processed" / "digital_twin_rl"

RECORD = {
    "format": "ovarian-twin-record/v1", "patient_label": "Test", "age_at_diagnosis": 60, "figo_stage": "IIIC",
    "tumor_grade": "G3", "residual_tumor_after_surgery": "1-10 mm",
    "treatments": [{"drugs": ["Carboplatin", "paclitaxel"], "start_month": 1.0, "end_month": 5.0}],
    "follow_ups": [{"month": 7.0, "tumor_status": "tumor-free"},
                   {"month": 20.0, "tumor_status": "tumor present", "event": "recurrence"}],
    "today_month": 21.0,
}


class RecordTests(unittest.TestCase):
    def test_record_becomes_a_pre_treatment_state(self):
        parsed = parse_record(RECORD)
        state = parsed["state"]
        self.assertEqual(state["line_number"], 2)
        self.assertEqual(state["setting"], "recurrent")          # derived from the last check-up event
        self.assertEqual(state["prior_exposure"], {"platinum": 1, "taxanes": 1})
        self.assertEqual(state["last_tumor_status"], "WT")
        self.assertAlmostEqual(state["months_since_last_platinum"], 16.0, delta=0.05)
        self.assertEqual(len(parsed["patient"]["lines"]), 1)

    def test_newly_diagnosed_without_treatments(self):
        record = {k: v for k, v in RECORD.items() if k not in ("treatments", "follow_ups")}
        state = parse_record(dict(record, today_month=1.0))["state"]
        self.assertEqual((state["line_number"], state["setting"]), (1, "primary"))

    def test_invalid_records_explain_the_problem(self):
        for change, message in [({"figo_stage": "V"}, "figo_stage"), ({"age_at_diagnosis": "sixty"}, "age_at_diagnosis"),
                                ({"today_month": 2.0}, "end_month")]:
            with self.assertRaisesRegex(RecordError, message):
                parse_record(dict(RECORD, **change))
        bad_drug = copy.deepcopy(RECORD)
        bad_drug["treatments"][0]["drugs"] = ["Aspirin"]
        with self.assertRaisesRegex(RecordError, "unknown drug"):
            parse_record(bad_drug)

    def test_demo_files_parse(self):
        files = sorted(SAMPLES.glob("patient_*.json"))
        self.assertGreaterEqual(len(files), 4)
        for path in files + [SAMPLES / "TEMPLATE.json"]:
            state = parse_record(json.loads(path.read_text(encoding="utf-8")))["state"]
            self.assertEqual(len(encode_state(state)), len(STATE_FEATURES))


@unittest.skipUnless((ARTIFACTS / "twins" / "twin_database.json").exists(), "trained artifacts not present")
class DemoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from demo.server import create_app
        cls.client = TestClient(create_app())

    def upload(self, name: str) -> dict:
        path = SAMPLES / name
        response = self.client.post("/api/twins", files={"file": (name, path.read_bytes(), "application/json")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_upload_builds_a_twin_from_five_neighbours_without_their_outcomes(self):
        twin = self.upload("patient_C_late_relapse.json")
        self.assertEqual(len(twin["build"]["neighbors"]), 5)
        text = json.dumps(twin["build"])
        for forbidden in ("pfs", "os_months", "response", "outcome", "died"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(twin["patient"]["situation"], "Recurrence")

    def test_predict_compare_simulate_and_ai_doctor(self):
        twin_id = self.upload("patient_A_newly_diagnosed.json")["id"]
        prediction = self.client.post(f"/api/twins/{twin_id}/predict", json={"medication": "platinum+taxanes"}).json()
        self.assertTrue(0 <= prediction["response_probability"] <= 1)
        self.assertGreater(prediction["expected_survival_months"], 0)
        self.assertTrue(prediction["side_effects"]["by_organ"])
        ranking = self.client.get(f"/api/twins/{twin_id}/compare").json()
        self.assertEqual(len(ranking), 14)
        simulation = self.client.post(f"/api/twins/{twin_id}/simulate",
                                      json={"medication": "platinum+taxanes", "runs": 30}).json()
        self.assertEqual(simulation["alive"][0], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(simulation["alive"], simulation["alive"][1:])))
        doctor = self.client.get(f"/api/twins/{twin_id}/ai-doctor").json()
        self.assertIn(doctor["choice_key"], [r["key"] for r in ranking])

    def test_guideline_constraint_for_platinum_resistant_patient(self):
        twin_id = self.upload("patient_D_early_relapse.json")["id"]
        flagged = self.client.post(f"/api/twins/{twin_id}/predict", json={"medication": "platinum+taxanes"}).json()
        self.assertTrue(flagged["guideline_flags"])
        unflagged = self.client.post(f"/api/twins/{twin_id}/predict", json={"medication": "anthracyclines"}).json()
        self.assertEqual(unflagged["guideline_flags"], [])
        doctor = self.client.get(f"/api/twins/{twin_id}/ai-doctor").json()
        self.assertNotIn("platinum", doctor["choice_key"])

    def test_bad_upload_and_unknown_medication(self):
        response = self.client.post("/api/twins", files={"file": ("x.json", b"{not json", "application/json")})
        self.assertEqual(response.status_code, 422)
        bad = json.dumps(dict(RECORD, figo_stage="V")).encode()
        response = self.client.post("/api/twins", files={"file": ("x.json", bad, "application/json")})
        self.assertEqual(response.status_code, 422)
        self.assertIn("figo_stage", response.json()["detail"])
        twin_id = self.upload("patient_D_early_relapse.json")["id"]
        response = self.client.post(f"/api/twins/{twin_id}/predict", json={"medication": "aspirin"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/twins/missing/compare").status_code, 404)


if __name__ == "__main__":
    unittest.main()
