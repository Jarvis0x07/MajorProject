"""Command line for the patient digital twin + RL doctor pipeline.

Run from the repository root:

  python -m digital_twin_rl all                       # every stage, end to end
  python -m digital_twin_rl data                      # treatment-line histories + action space
  python -m digital_twin_rl train-twin                # outcome-model ensemble + held-out report
  python -m digital_twin_rl twins                     # one twin card per patient
  python -m digital_twin_rl train-agent               # DQN and DQN+BCQ doctors inside the twins
  python -m digital_twin_rl evaluate                  # held-out policy comparison + figures
  python -m digital_twin_rl what-if --patient <id>    # every option for one patient
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import build_histories, load_histories, write_histories
from .dqn import DQNConfig, DQNPolicy, load_agent, save_agent, train_dqn
from .environment import RewardConfig, TwinTreatmentEnv, supported_regimens
from .evaluate import evaluate_policies, plot_results
from .outcome_model import OutcomeEnsemble, evaluate_ensemble, train_ensemble, write_json
from .policies import BehaviorModel, ClinicianPolicy, GuidelinePolicy, RandomPolicy
from .registry import build_registry, what_if
from .toxicity import ToxicityModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "clinical.project-tcga-ov.2026-10-09.json"
DEFAULT_OUT = ROOT / "data" / "processed" / "digital_twin_rl"
AGENTS = {"dqn": 0.0, "dqn_bcq": None}   # None -> --bcq-threshold


def member_split(n: int) -> tuple[list[int], list[int]]:
    """First half of the ensemble drives agent training; the second half is held out for evaluation."""
    if n < 4:
        return list(range(n)), list(range(n))
    return list(range(n // 2)), list(range(n // 2, n))


def load_context(out: Path) -> dict:
    histories = load_histories(out / "patient_histories.jsonl")
    regimens = [r["regimen"] for r in json.loads((out / "regimens.json").read_text(encoding="utf-8"))]
    ensemble = OutcomeEnsemble.load(out / "outcome_ensemble.pt")
    return {"histories": histories, "regimens": regimens, "ensemble": ensemble,
            "toxicity": ToxicityModel(), "behavior": BehaviorModel(histories, regimens)}


def load_agents(out: Path, bcq_threshold: float) -> dict:
    agents = {}
    for name, bcq in AGENTS.items():
        path = out / "agents" / f"{name}.pt"
        if path.exists():
            network, checkpoint = load_agent(path)
            agents[name] = (DQNPolicy(network, name), checkpoint.get("bcq_threshold", bcq_threshold if bcq is None else bcq))
    return agents


def cmd_data(args) -> None:
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    histories = build_histories(cases, args.seed)
    summary = write_histories(histories, args.out)
    regimens = supported_regimens(histories, args.min_support)
    write_json(args.out / "regimens.json", [{"regimen": r, "train_lines": n} for r, n in regimens])
    print(json.dumps({k: summary[k] for k in ("patients", "patients_with_lines", "lines", "response_labels",
                                              "pfs_events", "os_events", "patients_with_undated_given_treatment")}, indent=2))
    print(f"Action space: {len(regimens)} regimens with >= {args.min_support} training lines")


def cmd_train_twin(args) -> None:
    histories = load_histories(args.out / "patient_histories.jsonl")
    ensemble, member_reports = train_ensemble(histories, members=args.members, seed=args.seed)
    ensemble.save(args.out / "outcome_ensemble.pt")
    train_members, eval_members = member_split(len(ensemble))
    report = {
        "model": "multi-task MLP ensemble: response, progression-free and overall survival (discrete-time hazards)",
        "members": member_reports,
        "validation": evaluate_ensemble(ensemble, histories, "validation"),
        "test": evaluate_ensemble(ensemble, histories, "test"),
        "test_training_twin_members": evaluate_ensemble(ensemble, histories, "test", train_members),
        "test_evaluation_twin_members": evaluate_ensemble(ensemble, histories, "test", eval_members),
        "limits": [
            "Observational associations: treatment choice depends on unrecorded clinical factors (confounding by indication).",
            "Progression uses explicit progression/recurrence days, next-line start, or death; it is a real-world proxy, not RECIST.",
            "Side effects come from toxicity_profiles.json (assumed priors), not from this model.",
        ],
    }
    write_json(args.out / "outcome_model_report.json", report)
    test = report["test"]
    print(json.dumps({"test_response": test["response"], "test_pfs": test["progression_free_survival"],
                      "test_os": test["overall_survival"], "regimen_signal": test["regimen_signal"]}, indent=2))


def cmd_twins(args) -> None:
    ctx = load_context(args.out)
    result = build_registry(ctx["histories"], ctx["ensemble"], ctx["toxicity"], ctx["regimens"],
                            list(range(len(ctx["ensemble"]))), args.out / "twins" / "patient_twins.jsonl")
    print(json.dumps(result, indent=2))


def cmd_train_agent(args) -> None:
    ctx = load_context(args.out)
    train_members, _ = member_split(len(ctx["ensemble"]))
    histories_by_agent = {}
    for name, bcq in AGENTS.items():
        bcq = args.bcq_threshold if bcq is None else bcq
        print(f"Training {name} (BCQ threshold {bcq}) on training-split twins, ensemble members {train_members}")
        reward = RewardConfig(expected=True)
        env = TwinTreatmentEnv(ctx["histories"], ctx["ensemble"], ctx["toxicity"], ctx["regimens"], ctx["behavior"],
                               "train", train_members, reward, bcq, start="random", seed=args.seed)
        config = DQNConfig(episodes=args.episodes, seed=args.seed)
        network, history = train_dqn(env, config)
        save_agent(args.out / "agents" / f"{name}.pt", network, config, ctx["regimens"],
                   {"bcq_threshold": bcq, "train_members": train_members, "reward": reward.__dict__})
        histories_by_agent[name] = history
    write_json(args.out / "agents" / "training_history.json", histories_by_agent)


def cmd_evaluate(args) -> None:
    ctx = load_context(args.out)
    train_members, eval_members = member_split(len(ctx["ensemble"]))
    agents = load_agents(args.out, args.bcq_threshold)
    if not agents:
        raise SystemExit("No trained agents found; run `python -m digital_twin_rl train-agent` first")
    policies = {"random": (RandomPolicy(args.seed), 0.0), "clinician": (ClinicianPolicy(ctx["behavior"]), 0.0),
                "guideline": (GuidelinePolicy(ctx["regimens"]), 0.0), **agents}
    results = {}
    for twin_name, members in (("evaluation_twin", eval_members), ("training_twin", train_members)):
        for start in ("first", "relapse"):
            def factory(bcq, members=members, start=start):
                return TwinTreatmentEnv(ctx["histories"], ctx["ensemble"], ctx["toxicity"], ctx["regimens"],
                                        ctx["behavior"], args.split, members, RewardConfig(), bcq,
                                        start=start, seed=args.seed)
            key = f"{start}_line_{twin_name}"
            print(f"Evaluating {key} ...")
            results[key] = evaluate_policies(factory, policies, ctx["behavior"], args.rollouts, args.seed)
    write_json(args.out / "evaluation" / "policy_evaluation.json", {
        "split": args.split, "rollouts_per_patient": args.rollouts,
        "training_twin_members": train_members, "evaluation_twin_members": eval_members,
        "reward": RewardConfig().__dict__, "results": results,
    })
    history_path = args.out / "agents" / "training_history.json"
    training_history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    figures = plot_results({k: v for k, v in results.items() if k.endswith("evaluation_twin")},
                           training_history, args.out / "evaluation" / "figures")
    write_recommendations(ctx, agents, args)
    summary = {k: {p: {m: round(v["metrics"][m]["mean"], 3) for m in ("survival_months_5y", "first_line_response",
                                                                       "severe_side_effects", "clinician_supported_actions")}
                   for p, v in by_policy.items()} for k, by_policy in results.items()}
    print(json.dumps(summary, indent=2))
    print(f"Figures: {', '.join(figures)}")


def write_recommendations(ctx: dict, agents: dict, args) -> None:
    rows = []
    for patient in ctx["histories"]:
        if patient["split"] != args.split or not patient["lines"]:
            continue
        points = [0] + [next((l["line_index"] for l in patient["lines"]
                              if l["state"]["setting"] in ("recurrent", "progressive")), 0)]
        for line_index in sorted(set(points)):
            result = what_if(patient, line_index, ctx["ensemble"], ctx["toxicity"], ctx["regimens"],
                             list(range(len(ctx["ensemble"]))), ctx["behavior"], agents)
            options = {o["regimen"]: o for o in result["options"]}
            actual = result.get("what_actually_happened", {})
            row = {"patient_id": patient["patient_id"], "line": line_index + 1,
                   "setting": result["state"]["setting"], "actual_regimen": actual.get("regimen"),
                   "actual_predicted_survival_months_5y": actual.get("twin_prediction", {}).get("expected_survival_months_5y"),
                   "actual_observed_os_months_from_line": actual.get("observed", {}).get("os_months_from_line")}
            for name, regimen in result["recommendations"].items():
                row[f"{name}_regimen"] = regimen
                row[f"{name}_predicted_survival_months_5y"] = options.get(regimen, {}).get("expected_survival_months_5y")
            rows.append(row)
    if rows:
        path = args.out / "evaluation" / "test_patient_recommendations.csv"
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def cmd_what_if(args) -> None:
    ctx = load_context(args.out)
    matches = [p for p in ctx["histories"] if p["patient_id"].startswith(args.patient)]
    if len(matches) != 1:
        raise SystemExit(f"Patient id prefix {args.patient!r} matched {len(matches)} patients")
    result = what_if(matches[0], args.line - 1, ctx["ensemble"], ctx["toxicity"], ctx["regimens"],
                     list(range(len(ctx["ensemble"]))), ctx["behavior"], load_agents(args.out, args.bcq_threshold))
    print(json.dumps(result, indent=2))


def cmd_all(args) -> None:
    for step in (cmd_data, cmd_train_twin, cmd_twins, cmd_train_agent, cmd_evaluate):
        print(f"\n=== {step.__name__.removeprefix('cmd_').replace('_', '-')} ===")
        step(args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m digital_twin_rl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="TCGA-OV clinical JSON export")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support", type=int, default=10, help="training lines needed for a regimen to be an action")
    parser.add_argument("--members", type=int, default=10, help="outcome-model ensemble size")
    parser.add_argument("--episodes", type=int, default=30000, help="DQN training episodes per agent")
    parser.add_argument("--bcq-threshold", type=float, default=0.3, help="clinician-support threshold for dqn_bcq")
    parser.add_argument("--rollouts", type=int, default=30, help="evaluation rollouts per held-out patient")
    parser.add_argument("--split", default="test", choices=["validation", "test"], help="held-out patients to evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("data", cmd_data), ("train-twin", cmd_train_twin), ("twins", cmd_twins),
                       ("train-agent", cmd_train_agent), ("evaluate", cmd_evaluate), ("all", cmd_all)):
        sub.add_parser(name).set_defaults(func=func)
    what = sub.add_parser("what-if", help="compare every regimen for one patient")
    what.add_argument("--patient", required=True, help="patient id or unique prefix")
    what.add_argument("--line", type=int, default=1, help="observed treatment line to start from (1-based)")
    what.set_defaults(func=cmd_what_if)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
