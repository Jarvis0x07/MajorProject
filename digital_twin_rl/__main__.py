"""Run the pipeline from the project folder:

  python -m digital_twin_rl all            every step below, in order
  python -m digital_twin_rl prepare        1. raw clinical export -> patient treatment histories
  python -m digital_twin_rl train-model    2. outcome model, checked on held-out patients
  python -m digital_twin_rl build-twins    3. one digital twin per patient, from its 5 most similar patients
  python -m digital_twin_rl train-agent    4. DQN agent trained inside the twins, compared with other doctors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_histories, load_jsonl, save_jsonl
from .model import OutcomeModel, evaluate_model, train_model
from .rl import TwinEnv, compare_policies, save_agent, simulator_check, supported_regimens, train_agent
from .twin import NeighbourIndex, build_all_twins, check_twins

# ---------------------------------------------------------------------------------------------
# Where the input is read from and where every output is written.
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "clinical.project-tcga-ov.2026-10-09.json"
OUT = ROOT / "data" / "processed"
HISTORIES, REGIMENS, MODEL = OUT / "histories.jsonl", OUT / "regimens.json", OUT / "model.pt"
TWINS, AGENT, RESULTS = OUT / "twins.jsonl", OUT / "agent.pt", OUT / "results.json"


def save_result(key: str, value) -> None:
    # Add one result to results.json (keeping the others) and print it.
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    results[key] = value
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value}, indent=2))


# ---------------------------------------------------------------------------------------------
# The four steps.
# ---------------------------------------------------------------------------------------------
def prepare(args) -> None:
    # Step 1: raw export -> treatment histories, plus the list of regimens the agent can choose from.
    histories = build_histories(json.loads(RAW.read_text(encoding="utf-8")))
    save_jsonl(HISTORIES, histories)
    regimens = supported_regimens(histories)
    REGIMENS.write_text(json.dumps(regimens) + "\n", encoding="utf-8")
    print(f"{len(histories)} patients, {sum(len(p['lines']) for p in histories)} treatment lines, "
          f"{len(regimens)} regimens in the action space")


def train(args) -> None:
    # Step 2: train the outcome model, save it, and measure its accuracy on test patients.
    histories = load_jsonl(HISTORIES)
    model = train_model(histories)
    model.save(MODEL)
    save_result("outcome_model_on_test_patients", evaluate_model(model, histories, "test"))


def twins(args) -> None:
    # Step 3: one twin per patient from its 5 most similar patients; then check the twins' accuracy.
    histories, model = load_jsonl(HISTORIES), OutcomeModel.load(MODEL)
    index = NeighbourIndex(histories, model)
    records = build_all_twins(histories, model, index)
    save_jsonl(TWINS, records)
    print(f"{len(records)} twins written to {TWINS.relative_to(ROOT)}")
    save_result("twins_on_test_patients", check_twins(histories, model, index, "test"))


def agent(args) -> None:
    # Step 4: check the twins are a realistic environment, train the agent in the training
    # patients' twins, then compare it with other doctors in the test patients' twins.
    histories, model = load_jsonl(HISTORIES), OutcomeModel.load(MODEL)
    regimens = json.loads(REGIMENS.read_text(encoding="utf-8"))
    all_twins = load_jsonl(TWINS)
    print("Is the twin a realistic environment? (held-out patients, their real treatments replayed)")
    save_result("simulator_check", simulator_check(histories, all_twins, model, regimens))
    train_env = TwinEnv([t for t in all_twins if t["split"] == "train"], model, regimens)
    network, history = train_agent(train_env, args.episodes)
    save_agent(AGENT, network, regimens)
    test_env = TwinEnv([t for t in all_twins if t["split"] == "test"], model, regimens)
    save_result("learning_curve", history)
    save_result("doctors_on_test_twins", compare_policies(test_env, network))


STEPS = {"prepare": prepare, "train-model": train, "build-twins": twins, "train-agent": agent}


def main() -> None:
    # Read the command line, then run the chosen step (or all four in order).
    parser = argparse.ArgumentParser(prog="python -m digital_twin_rl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--episodes", type=int, default=30000, help="DQN training episodes (default 30000)")
    args = parser.parse_args()
    for name, step in STEPS.items():
        if args.step in (name, "all"):
            print(f"\n=== {name} ===")
            step(args)


if __name__ == "__main__":
    main()
