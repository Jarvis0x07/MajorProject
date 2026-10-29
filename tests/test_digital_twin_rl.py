import math
import random
import unittest

import numpy as np
import torch

from digital_twin_rl.data import build_patient, stage_number
from digital_twin_rl.dqn import DQNConfig, DQNPolicy, ReplayBuffer, train_dqn
from digital_twin_rl.environment import RewardConfig, TwinTreatmentEnv, safety_mask
from digital_twin_rl.outcome_model import (HORIZON_MONTHS, STATE_FEATURES, OutcomeEnsemble, OutcomeNet,
                                           discrete_survival_nll, encode_state, restricted_mean_months, time_bin)
from digital_twin_rl.policies import GuidelinePolicy
from digital_twin_rl.toxicity import ToxicityModel
from digital_twin_rl.twin import PatientDigitalTwin, sample_event_time
from scripts.paper_method_pipeline import ACTION_CLASSES

REGIMENS = [["platinum", "taxanes"], ["platinum"], ["anthracyclines"], ["topoisomerase_inhibitors"],
            ["taxanes"], ["hormonal_agents"]]


def treatment(agent, start, end=None, status=None, given="yes", ttype="Chemotherapy"):
    return {"therapeutic_agents": agent, "treatment_type": ttype, "treatment_or_therapy": given,
            "days_to_treatment_start": start, "days_to_treatment_end": end, "initial_disease_status": status}


CASE = {
    "case_id": "patient-a", "index_date": "Diagnosis",
    "demographic": {"age_at_index": 60, "days_to_death": 900, "vital_status": "Dead"},
    "diagnoses": [{
        "diagnosis_is_primary_disease": "true", "figo_stage": "Stage IIIC", "tumor_grade": "G3",
        "pathology_details": [{"residual_tumor_measurement": "1-10 mm"}],
        "treatments": [
            treatment("Carboplatin", 20, 150), treatment("Paclitaxel", 20, 150),
            treatment(None, None, given="no", ttype="Radiation Therapy, NOS"),
            treatment("Topotecan", 400, 500, status="Progressive Disease"),
        ],
    }],
    "follow_ups": [
        {"days_to_follow_up": 200, "disease_response": "TF-Tumor Free"},
        {"days_to_follow_up": 380, "days_to_recurrence": 380, "disease_response": "WT-With Tumor"},
    ],
}


def tiny_ensemble(members=2, seed=0):
    torch.manual_seed(seed)
    models = [OutcomeNet(len(STATE_FEATURES), len(ACTION_CLASSES), hidden=8).eval() for _ in range(members)]
    return OutcomeEnsemble(models, hidden=8)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.patient = build_patient(CASE, "train")

    def test_not_given_records_are_excluded_and_lines_grouped(self):
        self.assertEqual(self.patient["undated_given_treatments"], 0)
        lines = self.patient["lines"]
        self.assertEqual([l["regimen"] for l in lines], [["platinum", "taxanes"], ["topoisomerase_inhibitors"]])
        self.assertEqual(lines[0]["state"]["setting"], "primary")
        self.assertEqual(lines[1]["state"]["setting"], "progressive")

    def test_outcomes_use_follow_up_progression_and_death(self):
        first, second = self.patient["lines"]
        self.assertEqual(first["outcome"]["response"], 1)
        self.assertEqual(first["outcome"]["pfs_event"], 1)
        self.assertAlmostEqual(first["outcome"]["pfs_months"], (380 - 20) / 30.44, places=3)
        self.assertEqual(first["outcome"]["os_event"], 1)
        self.assertAlmostEqual(second["outcome"]["os_months"], (900 - 400) / 30.44, places=3)

    def test_second_line_state_tracks_prior_platinum(self):
        state = self.patient["lines"][1]["state"]
        self.assertEqual(state["prior_exposure"], {"platinum": 1, "taxanes": 1})
        self.assertAlmostEqual(state["months_since_last_platinum"], (400 - 150) / 30.44, places=3)
        self.assertEqual(state["last_tumor_status"], "WT")
        self.assertEqual(len(encode_state(state)), len(STATE_FEATURES))

    def test_stage_parsing(self):
        self.assertEqual([stage_number(s) for s in ("Stage IIIC", "Stage IV", "Stage IC", "Stage IIA", None)],
                         [3, 4, 1, 2, None])


class OutcomeModelTests(unittest.TestCase):
    def test_discrete_survival_likelihood(self):
        logits = torch.zeros(1, 6)          # every hazard 0.5
        event = discrete_survival_nll(logits, torch.tensor([1]), torch.tensor([1.0]))
        censored = discrete_survival_nll(logits, torch.tensor([1]), torch.tensor([0.0]))
        self.assertAlmostEqual(float(event), -2 * math.log(0.5), places=5)
        self.assertAlmostEqual(float(censored), -math.log(0.5), places=5)

    def test_time_bins_and_restricted_mean(self):
        self.assertEqual([time_bin(m) for m in (0, 2.9, 3, 12, 47, 100)], [0, 0, 1, 3, 4, 5])
        self.assertAlmostEqual(float(restricted_mean_months(np.zeros(6))), HORIZON_MONTHS)
        self.assertAlmostEqual(float(restricted_mean_months(np.zeros(6), 10.0)), 10.0)
        certain_early = np.array([1.0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(float(restricted_mean_months(certain_early)), 1.5)

    def test_event_time_sampling(self):
        self.assertIsNone(sample_event_time(np.zeros(6), 0.5))
        t = sample_event_time(np.array([1.0, 0, 0, 0, 0, 0]), 0.5)
        self.assertTrue(0.0 <= t <= 3.0)


class ToxicityAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tox = ToxicityModel()

    def test_risk_reflects_class_profiles_and_patient_modifiers(self):
        young = {"age_years": 55, "prior_exposure": {}}
        old = {"age_years": 78, "prior_exposure": {}}
        taxane = self.tox.risk(young, ["taxanes"])["severe_event_probability"]["neuropathy"]
        platinum = self.tox.risk(young, ["platinum"])["severe_event_probability"]["neuropathy"]
        self.assertGreater(taxane, platinum)
        self.assertGreater(self.tox.risk(old, ["taxanes"])["probability_any_severe"],
                           self.tox.risk(young, ["taxanes"])["probability_any_severe"])
        self.assertEqual(self.tox.risk(young, ["hormonal_agents"])["levels"]["hematologic"], "none")
        for p in self.tox.risk(old, ["platinum", "taxanes"])["severe_event_probability"].values():
            self.assertLessEqual(p, self.tox.max_probability)

    def test_safety_rules(self):
        mask = safety_mask({"severe_toxicity_history": {"neuropathy": 1}}, REGIMENS)
        self.assertFalse(mask[0])
        self.assertFalse(mask[4])
        self.assertTrue(mask[1])
        poor = safety_mask({"ecog": 3}, REGIMENS)
        self.assertFalse(poor[0])
        self.assertTrue(poor.any())


class TwinEnvironmentTests(unittest.TestCase):
    def setUp(self):
        patient = build_patient(CASE, "train")
        self.histories = [patient, dict(patient, patient_id="patient-b")]
        self.ensemble = tiny_ensemble()
        self.tox = ToxicityModel()

    def test_twin_predicts_and_steps(self):
        twin = PatientDigitalTwin(self.histories[0], self.ensemble, self.tox, start_line=1)
        prediction = twin.predict(["platinum", "taxanes"])
        self.assertTrue(0.0 <= prediction["response_probability"] <= 1.0)
        self.assertIn("neuropathy", prediction["side_effects"]["levels"])
        outcome = twin.step(["platinum"], random.Random(0))
        self.assertGreaterEqual(outcome["months_on_line"], 0.0)
        if outcome["progressed"]:
            self.assertEqual(twin.state["line_number"], 3)

    def test_episode_runs_and_masked_actions_are_rejected(self):
        env = TwinTreatmentEnv(self.histories, self.ensemble, self.tox, REGIMENS, start="random",
                               reward=RewardConfig(), seed=3)
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, (env.observation_dim,))
        done, steps = False, 0
        while not done:
            action = int(np.flatnonzero(info["action_mask"])[0])
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(np.isfinite(reward))
            done, steps = terminated or truncated, steps + 1
        self.assertLessEqual(steps, env.max_lines)
        env.reset(seed=2)
        env.twin.state["severe_toxicity_history"] = {"neuropathy": 1}
        with self.assertRaises(ValueError):
            env.step(0)

    def test_dqn_trains_and_respects_mask(self):
        env = TwinTreatmentEnv(self.histories, self.ensemble, self.tox, REGIMENS, start="random", seed=0)
        network, history = train_dqn(env, DQNConfig(episodes=30, warmup_steps=5, batch_size=8), log_every=10)
        self.assertEqual(len(history), 3)
        mask = np.zeros(len(REGIMENS), dtype=bool)
        mask[3] = True
        obs, _ = env.reset(seed=0)
        self.assertEqual(DQNPolicy(network).act(obs, {"action_mask": mask}), 3)

    def test_n_step_returns_bootstrap_only_from_non_terminal_states(self):
        buffer = ReplayBuffer(10, 1, 2)
        obs = [np.array([float(i)], dtype=np.float32) for i in range(4)]
        mask = np.ones(2, dtype=bool)
        buffer.add_episode([(obs[0], 0, 1.0, obs[1], False, mask), (obs[1], 1, 2.0, obs[2], False, mask),
                            (obs[2], 0, 3.0, obs[3], True, mask)], gamma=0.5, n_step=2)
        self.assertEqual(buffer.size, 3)
        np.testing.assert_allclose(buffer.reward[:3], [2.0, 3.5, 3.0])
        np.testing.assert_allclose(buffer.discount[:3], [0.25, 0.25, 0.5])
        np.testing.assert_allclose(buffer.done[:3], [0.0, 1.0, 1.0])
        self.assertEqual(float(buffer.next_obs[0][0]), 2.0)


class GuidelinePolicyTests(unittest.TestCase):
    def test_first_line_and_platinum_resistant_relapse(self):
        policy = GuidelinePolicy(REGIMENS)
        mask = np.ones(len(REGIMENS), dtype=bool)
        first = policy.act(None, {"state": {"setting": "primary", "line_number": 1}, "action_mask": mask})
        self.assertEqual(REGIMENS[first], ["platinum", "taxanes"])
        resistant = {"setting": "progressive", "line_number": 2, "months_since_last_platinum": 2.0,
                     "prior_exposure": {"platinum": 1, "anthracyclines": 1}}
        choice = policy.act(None, {"state": resistant, "action_mask": mask})
        self.assertEqual(REGIMENS[choice], ["topoisomerase_inhibitors"])


if __name__ == "__main__":
    unittest.main()
