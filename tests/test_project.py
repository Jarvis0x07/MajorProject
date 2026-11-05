import copy
import json
import random
import unittest
from pathlib import Path

from digital_twin_rl.data import build_patient
from digital_twin_rl.model import FEATURES, encode_state
from digital_twin_rl.record import RecordError, parse_record

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "demo" / "sample_patients"
TRAINED = (ROOT / "data" / "processed" / "agent.pt").exists()

RECORD = {
    "age_at_diagnosis": 60, "figo_stage": "IIIC", "tumor_grade": "G3", "residual_tumor_after_surgery": "1-10 mm",
    "treatments": [{"drugs": ["Carboplatin", "paclitaxel"], "start_month": 1.0, "end_month": 5.0}],
    "follow_ups": [{"month": 20.0, "event": "recurrence"}],
    "today_month": 21.0,
}


class DataTests(unittest.TestCase):
    def test_treatment_lines_and_outcomes(self):
        case = {"case_id": "p1", "demographic": {"age_at_index": 55, "vital_status": "Dead", "days_to_death": 900},
                "diagnoses": [{"diagnosis_is_primary_disease": "true", "figo_stage": "Stage IIIC", "treatments": [
                    {"therapeutic_agents": "Carboplatin", "days_to_treatment_start": 30},
                    {"therapeutic_agents": "Paclitaxel", "days_to_treatment_start": 40},       # same line (< 6 weeks)
                    {"treatment_type": "Radiation Therapy, NOS", "treatment_or_therapy": "no",
                     "days_to_treatment_start": 100},                                          # not given: ignored
                    {"therapeutic_agents": "Topotecan", "days_to_treatment_start": 400,
                     "initial_disease_status": "Recurrent Disease"}]}]}
        lines = build_patient(case, "train")["lines"]
        self.assertEqual([l["regimen"] for l in lines], [["platinum", "taxanes"], ["topoisomerase_inhibitors"]])
        self.assertEqual(lines[1]["state"]["setting"], "recurrent")
        self.assertEqual(lines[1]["state"]["prior_exposure"], {"platinum": 1, "taxanes": 1})
        first = lines[0]["outcome"]
        self.assertAlmostEqual(first["pfs_months"], (400 - 30) / 30.44)        # next line started
        self.assertEqual((first["os_event"], lines[1]["outcome"]["pfs_event_is_death"]), (1, 1))


class RecordTests(unittest.TestCase):
    def test_record_becomes_a_state(self):
        parsed = parse_record(RECORD)
        state = parsed["state"]
        self.assertEqual((state["line_number"], state["setting"]), (2, "recurrent"))
        self.assertEqual(state["prior_exposure"], {"platinum": 1, "taxanes": 1})
        self.assertAlmostEqual(state["months_since_last_platinum"], 16.0, delta=0.05)
        self.assertEqual(len(parsed["past_lines"]), 1)

    def test_newly_diagnosed(self):
        state = parse_record({k: v for k, v in RECORD.items() if k not in ("treatments", "follow_ups")})["state"]
        self.assertEqual((state["line_number"], state["setting"]), (1, "primary"))

    def test_invalid_records_explain_the_problem(self):
        for change, message in [({"figo_stage": "V"}, "figo_stage"), ({"age_at_diagnosis": "sixty"}, "age_at_diagnosis"),
                                ({"today_month": 2.0}, "end_month")]:
            with self.assertRaisesRegex(RecordError, message):
                parse_record(dict(RECORD, **change))
        bad = copy.deepcopy(RECORD)
        bad["treatments"][0]["drugs"] = ["Aspirin"]
        with self.assertRaisesRegex(RecordError, "unknown drug"):
            parse_record(bad)

    def test_sample_files_parse(self):
        for path in SAMPLES.glob("*.json"):
            state = parse_record(json.loads(path.read_text(encoding="utf-8")))["state"]
            self.assertEqual(len(encode_state(state)), len(FEATURES), path.name)


@unittest.skipUnless(TRAINED, "run `python -m digital_twin_rl all` first")
class TrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from demo.server import Lab, create_app
        cls.lab = Lab()
        cls.client = TestClient(create_app(cls.lab))

    def upload(self, name: str, content: bytes | None = None):
        content = content if content is not None else (SAMPLES / name).read_bytes()
        return self.client.post("/api/analyse", files={"file": (name, content, "application/json")})

    def test_twin_is_built_from_five_other_patients_and_can_be_simulated(self):
        from digital_twin_rl.twin import build_twin
        state = parse_record(RECORD)["state"]
        twin = build_twin(state, self.lab.model, self.lab.index, exclude="someone")
        self.assertEqual(len({n["patient_id"] for n in twin.neighbours}), 5)
        prediction = twin.predict(["platinum", "taxanes"])
        self.assertTrue(0 < prediction["response"] < 1 and 0 < prediction["survival_months"] <= 60)
        rng = random.Random(0)
        while not twin.ended:
            twin.step(["platinum", "taxanes"], rng)
        self.assertLessEqual(twin.months_elapsed, 60.0 + 1e-6)

    def test_environment_masks_platinum_when_resistant(self):
        from digital_twin_rl.rl import allowed
        state = parse_record(json.loads((SAMPLES / "patient_D_early_relapse.json").read_text()))["state"]
        mask = allowed(state, self.lab.regimens)
        self.assertFalse(any(ok for r, ok in zip(self.lab.regimens, mask) if "platinum" in r))
        self.assertTrue(mask.any())

    def test_website_ranks_every_medication(self):
        response = self.upload("patient_C_late_relapse.json")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        months = [m["survival_months"] for m in result["medications"]]
        self.assertEqual(len(months), len(self.lab.regimens))
        self.assertEqual(months, sorted(months, reverse=True))
        self.assertIn(result["agent_choice"], [m["key"] for m in result["medications"]])
        self.assertEqual(len(result["neighbours"]), 5)
        for forbidden in ("os_", "pfs", "patient_id"):            # neighbours' outcomes and ids stay private
            self.assertNotIn(forbidden, json.dumps(result["neighbours"]))

    def test_website_flags_platinum_for_resistant_patient(self):
        result = self.upload("patient_D_early_relapse.json").json()
        self.assertTrue(result["patient"]["platinum_resistant"])
        flagged = {m["key"] for m in result["medications"] if m["guideline_caution"]}
        self.assertEqual(flagged, {m["key"] for m in result["medications"] if "platinum" in m["key"]})
        self.assertNotIn("platinum", result["agent_choice"])

    def test_website_rejects_bad_files(self):
        self.assertEqual(self.upload("x.json", b"{not json").status_code, 422)
        response = self.upload("x.json", json.dumps(dict(RECORD, figo_stage="V")).encode())
        self.assertEqual(response.status_code, 422)
        self.assertIn("figo_stage", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
