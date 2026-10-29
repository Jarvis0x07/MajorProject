"""Patient digital twins and an RL treatment agent for TCGA-OV ovarian cancer.

Pipeline: treatment-line histories (data) -> outcome-model ensemble
(outcome_model) -> per-patient twins (twin) -> twin-backed environment
(environment) -> DQN doctor agent (dqn) -> policy evaluation (evaluate).
Run `python -m digital_twin_rl --help` from the repository root.

Research prototype on retrospective observational data. Not a clinical tool.
"""
