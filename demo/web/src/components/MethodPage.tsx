import type { ReactNode } from "react";
import { motion } from "framer-motion";
import type { ProjectInfo } from "../api";
import { pct } from "../format";
import { LineChart } from "./LineChart";
import { Card, Heading, Label, Stat } from "./primitives";

function Objective({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }} transition={{ duration: 0.35 }}>
      <Card>
        <div className="flex items-baseline gap-4">
          <span className="font-display text-5xl font-semibold text-teal/80">{number}</span>
          <div>
            <Label>Objective {number}</Label>
            <Heading as="h2">{title}</Heading>
          </div>
        </div>
        <div className="mt-5">{children}</div>
      </Card>
    </motion.div>
  );
}

export function MethodPage({ info }: { info: ProjectInfo }) {
  const acc = info.twin_accuracy;
  const rl = info.rl.first_line_survival_months;
  const baselines: [string, number][] = [
    ["Random choices", rl.random],
    ["Copy usual practice", rl.clinician],
    ["Simplified guideline", rl.guideline],
    ["Basic RL agent (DQN)", rl.dqn],
  ];
  const maxBaseline = Math.max(...baselines.map(([, v]) => v));

  return (
    <div className="space-y-6">
      <Objective number={1} title="A digital twin for every patient">
        <p className="max-w-3xl text-sm leading-relaxed text-ink-soft">
          Each patient's history is split into treatment lines. A 10-network ensemble learns, from the patient's situation before a
          line and the drugs given, the chance the tumour disappears, the time until the cancer returns or grows (and whether
          that event is death), and survival. A twin is that model evaluated at one patient's own data. New patients get a twin
          on the spot, calibrated with the 5 most similar of {info.cohort.stored_twins.toLocaleString()} stored twins.
        </p>
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Patients (TCGA-OV)" value={info.cohort.patients} detail={`${info.cohort.patients_with_lines} with dated treatment lines`} />
          <Stat label="Treatment lines learned from" value={info.cohort.treatment_lines.toLocaleString()} />
          <Stat label="Response prediction (AUROC)" value={acc.response_auroc.toFixed(2)} detail="unseen test patients; 0.5 = coin flip" />
          <Stat label="Survival ranking (C-index)" value={acc.survival_c_index.toFixed(2)} detail={`1-year mortality ${pct(acc.mortality_12m_predicted, 1)} predicted vs ${pct(acc.mortality_12m_observed, 1)} observed`} />
        </div>
        <p className="mt-4 text-sm text-ink-soft">
          Calibrating with the 5 nearest twins changed the 12-month progression error (Brier score) on validation patients from{" "}
          <span className="num text-ink">{info.neighbor_twins.validation_progression_brier_shared.toFixed(4)}</span> to{" "}
          <span className="num text-ink">{info.neighbor_twins.validation_progression_brier_neighbor.toFixed(4)}</span>: a small
          gain. Adjusting survival or response from only five patients made predictions worse, so those come from the shared model.
        </p>
      </Objective>

      <Objective number={2} title="The twin as a simulator, checked against reality">
        <p className="max-w-3xl text-sm leading-relaxed text-ink-soft">
          The twin plays a patient forward line by line: side effects, response, then progression or death, after which the next
          decision comes. To test it, every training patient's real treatments were replayed in their twin and the simulated
          survival compared with what actually happened ({info.simulation_fidelity.patients} patients).
        </p>
        <div className="mt-4 max-w-3xl">
          <LineChart
            x={info.simulation_fidelity.months}
            xLabel="Months from first treatment"
            yLabel="Proportion alive"
            formatX={(v) => `${v}`}
            series={[
              { label: "Observed (Kaplan-Meier)", values: info.simulation_fidelity.observed, color: "var(--color-ink)", markers: true },
              { label: "Twins replaying the same treatments", values: info.simulation_fidelity.simulated, color: "var(--color-teal)", dashed: true, markers: true },
            ]}
            summary="Simulated and observed survival curves overlap closely for four years."
          />
        </div>
      </Objective>

      <Objective number={3} title="Reinforcement-learning framework">
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <dl className="space-y-2 text-sm">
              {[
                ["State", "37 features: age, stage, grade, residual tumour, line, situation, past drugs, platinum-free interval"],
                ["Action", "One of 14 regimens for the next treatment line"],
                ["Transition", "The patient's twin simulates that line"],
                ["Reward", "Years alive + bonus if the tumour disappears − severe side effects − twin uncertainty"],
                ["Constraints", "Only regimens seen ≥10 times; safety rules after severe toxicity; guideline rule against platinum in platinum-resistant relapse"],
                ["Episode", "Until death, long remission, 5 years or 8 lines"],
              ].map(([term, text]) => (
                <div key={term} className="grid grid-cols-[6.5rem_1fr] gap-3 border-b border-line-soft pb-2">
                  <dt className="font-medium text-ink">{term}</dt>
                  <dd className="text-ink-soft">{text}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-5 text-sm font-medium text-ink">Basic check: does training improve the agent?</p>
            <LineChart
              x={info.rl.learning_curve.map((p) => p.episode)}
              series={[{ label: "Average reward per patient journey", values: info.rl.learning_curve.map((p) => p.mean_return), color: "var(--color-teal)" }]}
              xLabel="Training episodes (simulated patients treated)"
              yLabel="Reward"
              yMin={Math.floor(Math.min(...info.rl.learning_curve.map((p) => p.mean_return)) * 5) / 5}
              yMax={Math.ceil(Math.max(...info.rl.learning_curve.map((p) => p.mean_return)) * 5) / 5}
              formatY={(v) => v.toFixed(1)}
              formatX={(v) => `${Math.round(v / 1000)}k`}
              height={220}
              summary="The agent's average reward rises during training, showing the framework learns."
            />
          </div>
          <div>
            <p className="text-sm font-medium text-ink">Simulated months alive within 5 years, from first treatment</p>
            <p className="mt-1 text-xs text-muted">{info.rl.test_patients} unseen test patients, scored in twins the agent never trained in</p>
            <ul className="mt-4 space-y-3">
              {baselines.map(([name, value], i) => (
                <li key={name}>
                  <div className="flex justify-between text-sm">
                    <span className="text-ink-soft">{name}</span>
                    <span className="num text-ink">{value.toFixed(1)}</span>
                  </div>
                  <div className="mt-1 h-2.5 overflow-hidden rounded-full bg-line-soft">
                    <motion.div
                      className={`h-full rounded-full ${i === baselines.length - 1 ? "bg-teal" : "bg-ink/40"}`}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(value / maxBaseline) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.6, delay: i * 0.08 }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Objective>

      <Card className="bg-paper/80">
        <Heading as="h3">Limits to keep in mind</Heading>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-soft">
          <li>Learned from historical, observational records: the twin captures associations, not proven cause and effect.</li>
          <li>Side-effect risks are drug-class assumptions; the dataset records no side effects, doses, costs or quality of life.</li>
          <li>All agent results come from simulation. Safer RL (Objective 4) and full evaluation (Objective 5) are next semester's work.</li>
          <li>Research prototype on public, de-identified TCGA data. Not medical advice.</li>
        </ul>
      </Card>
    </div>
  );
}
