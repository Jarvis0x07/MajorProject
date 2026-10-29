"""Per-patient twin registry and what-if queries.

`build_registry` instantiates one twin per eligible patient and stores a
"twin card": the patient's starting state, observed treatment history, and the
twin's predicted outcomes for every regimen in the action space, including
the regimen the patient actually received.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .environment import action_mask, build_observation, safety_mask
from .twin import PatientDigitalTwin


def compact(prediction: dict) -> dict:
    side = prediction["side_effects"]
    return {
        "regimen": prediction["regimen"],
        "response_probability": round(prediction["response_probability"], 3),
        "progression_free_12m": round(prediction["progression_free_probability"]["12m"], 3),
        "expected_progression_free_months_5y": round(prediction["expected_progression_free_months_5y"], 1),
        "survival_12m": round(prediction["survival_probability"]["12m"], 3),
        "expected_survival_months_5y": round(prediction["expected_survival_months_5y"], 1),
        "severe_side_effect_probability": round(side["probability_any_severe"], 3),
        "main_side_effect_risks": {o: l for o, l in side["levels"].items() if l in ("moderate", "high")},
        "survival_months_std": round(prediction["uncertainty"]["survival_months_std"], 2),
    }


def observed_history(patient: dict) -> list[dict]:
    return [{"line": l["line_index"] + 1, "start_day": l["start_day"], "setting": l["state"]["setting"],
             "regimen": "+".join(l["regimen"]), "response": l["outcome"]["response"],
             "pfs_months": round(l["outcome"]["pfs_months"], 1), "pfs_event": l["outcome"]["pfs_event"],
             "os_months_from_line": round(l["outcome"]["os_months"], 1), "os_event": l["outcome"]["os_event"]}
            for l in patient["lines"]]


def twin_card(patient: dict, ensemble, toxicity, regimens: list[list[str]], members: list[int]) -> dict:
    twin = PatientDigitalTwin(patient, ensemble, toxicity, 0, members)
    options = [compact(twin.predict(r)) for r in regimens]
    options.sort(key=lambda o: -o["expected_survival_months_5y"])
    card = {
        "patient_id": patient["patient_id"], "split": patient["split"],
        "vital_status": patient.get("vital_status"), "static": patient["static"],
        "twin_start": twin.state_source,
        "starting_state": {k: v for k, v in twin.state.items() if k != "severe_toxicity_history"},
        "observed_history": observed_history(patient),
        "what_if_by_regimen": options,
    }
    if patient["lines"]:
        actual = patient["lines"][0]
        card["actual_first_regimen"] = {"twin_prediction": compact(twin.predict(actual["regimen"])),
                                        "observed_outcome": observed_history(patient)[0]}
    return card


def build_registry(histories: list[dict], ensemble, toxicity, regimens, members, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for patient in histories:
            stream.write(json.dumps(twin_card(patient, ensemble, toxicity, regimens, members)) + "\n")
            count += 1
    return {"twins_written": count, "path": str(out_path)}


def what_if(patient: dict, line_index: int, ensemble, toxicity, regimens, members,
            behavior, agents: dict) -> dict:
    """All regimen options for one patient at one decision point, with each doctor's choice.

    agents: name -> (policy, bcq_threshold the policy was trained with).
    """
    twin = PatientDigitalTwin(patient, ensemble, toxicity, line_index, members)
    mask = safety_mask(twin.state, regimens)
    probs = behavior.probabilities(twin.state)
    options = []
    for i, regimen in enumerate(regimens):
        option = compact(twin.predict(regimen))
        option["allowed_by_safety_rules"] = bool(mask[i])
        option["clinician_choice_probability"] = round(float(probs[i]), 3)
        options.append(option)
    options.sort(key=lambda o: -o["expected_survival_months_5y"])
    choices = {}
    observation = build_observation(twin, toxicity.organs)
    for name, (policy, bcq_threshold) in agents.items():
        info = {"state": twin.state, "action_mask": action_mask(twin.state, regimens, behavior, bcq_threshold)}
        choices[name] = "+".join(regimens[policy.act(observation, info)])
    choices["clinician_model"] = "+".join(regimens[int(np.argmax(np.where(mask, probs, -1)))])
    result = {"patient_id": patient["patient_id"], "decision_point": twin.state_source,
              "state": {k: v for k, v in twin.state.items() if k != "severe_toxicity_history"},
              "personal_calibration_offsets": twin.offsets, "recommendations": choices, "options": options}
    if patient["lines"]:
        line = patient["lines"][min(line_index, len(patient["lines"]) - 1)]
        result["what_actually_happened"] = {"regimen": "+".join(line["regimen"]),
                                            "twin_prediction": compact(twin.predict(line["regimen"])),
                                            "observed": observed_history(patient)[line["line_index"]]}
    return result
