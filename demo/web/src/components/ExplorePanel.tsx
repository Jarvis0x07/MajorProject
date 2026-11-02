import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bot, ChartSpline, ListOrdered, TriangleAlert } from "lucide-react";
import { api, type AiDoctor, type ComparisonRow, type Simulation } from "../api";
import { pct } from "../format";
import { LineChart } from "./LineChart";
import { Button, Card, Heading, Label, Stat } from "./primitives";

type Tab = "simulate" | "compare" | "doctor";

interface Props {
  twinId: string;
  medication: string;
  medicationName: string;
  comparison: ComparisonRow[] | null;
  onPick: (key: string) => void;
}

export function ExplorePanel({ twinId, medication, medicationName, comparison, onPick }: Props) {
  const [tab, setTab] = useState<Tab>("simulate");
  const [simulation, setSimulation] = useState<Simulation | null>(null);
  const [doctor, setDoctor] = useState<AiDoctor | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSimulation(null);
  }, [twinId, medication]);
  useEffect(() => {
    setDoctor(null);
  }, [twinId]);

  const run = async <T,>(work: () => Promise<T>, done: (value: T) => void) => {
    setBusy(true);
    setError(null);
    try {
      done(await work());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const tabs: { id: Tab; label: string; icon: typeof Bot }[] = [
    { id: "simulate", label: "Simulate next 5 years", icon: ChartSpline },
    { id: "compare", label: "Compare all medications", icon: ListOrdered },
    { id: "doctor", label: "Ask the AI doctor", icon: Bot },
  ];

  return (
    <Card>
      <Label>Step 5 · Explore</Label>
      <div className="mt-3 flex flex-wrap gap-2" role="tablist" aria-label="Explore the twin">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={`inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-xl border px-4 text-sm font-medium transition-colors duration-200 ${
              tab === id ? "border-ink bg-ink text-paper" : "border-line bg-surface text-ink hover:border-ink/40"
            }`}
          >
            <Icon className="size-4" aria-hidden /> {label}
          </button>
        ))}
      </div>

      {error && <p role="alert" className="mt-4 text-sm text-coral">{error}</p>}

      <div className="mt-6" role="tabpanel">
        {tab === "simulate" && (
          <div>
            <Heading as="h3">Play the twin forward: {medicationName} now, then standard care</Heading>
            <p className="mt-1 text-sm text-ink-soft">
              The twin simulates many possible futures, one treatment line at a time: side effects, response, progression or
              death. Later lines follow a simplified guideline. This is Objective 2: the twin used as a simulator.
            </p>
            {!simulation && (
              <Button variant="primary" className="mt-4" busy={busy} onClick={() => run(() => api.simulate(twinId, medication), setSimulation)}>
                Run 300 simulated futures
              </Button>
            )}
            {simulation && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-5">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Stat label="Median survival" value={simulation.median_survival_months ? `${simulation.median_survival_months} mo` : "> 60 mo"} />
                  <Stat label="Alive at 5 years" value={pct(simulation.alive_at_5_years)} />
                  <Stat label="Treatment lines used" value={simulation.mean_treatment_lines.toFixed(1)} detail="average" />
                  <Stat label="Severe side effects" value={simulation.mean_severe_side_effects.toFixed(1)} detail="average over 5 years" />
                </div>
                <div className="mt-5">
                  <LineChart
                    x={simulation.months}
                    xLabel="Months from today"
                    yLabel="Proportion of simulated futures"
                    formatX={(v) => `${v}`}
                    series={[
                      { label: "Alive", values: simulation.alive, color: "var(--color-teal)" },
                      { label: `Cancer controlled on ${medicationName}`, values: simulation.progression_free_first_line, color: "var(--color-ink)", dashed: true },
                    ]}
                    summary={`Simulated survival over 60 months with ${medicationName}: ${pct(simulation.alive_at_5_years)} alive at 5 years.`}
                  />
                </div>
                {simulation.example_journeys[0] && (
                  <div className="mt-5">
                    <p className="text-sm font-medium text-ink">One simulated future, line by line</p>
                    <ol className="mt-2 flex flex-wrap items-stretch gap-2">
                      {simulation.example_journeys[0].map((step, i) => (
                        <li key={i} className="min-w-40 flex-1 rounded-xl border border-line-soft bg-paper/60 p-3 text-sm">
                          <p className="font-medium text-ink">{step.medication}</p>
                          <p className="num mt-1 text-xs text-ink-soft">{step.months} months · {step.response ? "tumour cleared" : "tumour remained"}</p>
                          <p className="mt-1 text-xs text-muted">
                            {step.severe_side_effects.length ? `severe: ${step.severe_side_effects.join(", ")}` : "no severe side effects"}
                          </p>
                          <p className="mt-1 text-xs text-ink-soft">then: {step.ended_with.replaceAll("_", " ")}</p>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        )}

        {tab === "compare" && (
          <div>
            <Heading as="h3">All 14 options for this patient's twin</Heading>
            <p className="mt-1 text-sm text-ink-soft">Sorted by expected months alive within 5 years. Select a row to inspect it.</p>
            {!comparison ? (
              <p className="mt-4 text-sm text-muted">Loading…</p>
            ) : (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[40rem] text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-muted">
                      <th className="py-2 pr-3 font-medium">Medication</th>
                      <th className="py-2 pr-3 text-right font-medium">Tumour disappears</th>
                      <th className="py-2 pr-3 text-right font-medium">Months controlled</th>
                      <th className="py-2 pr-3 font-medium">Months alive (5 y)</th>
                      <th className="py-2 text-right font-medium">Severe side effect</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.map((r) => {
                      const max = comparison[0].expected_survival_months;
                      return (
                        <tr
                          key={r.key}
                          onClick={() => onPick(r.key)}
                          className={`cursor-pointer border-b border-line-soft transition-colors hover:bg-teal-soft/40 ${r.key === medication ? "bg-teal-soft/60" : ""}`}
                        >
                          <td className="py-2.5 pr-3">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onPick(r.key);
                              }}
                              className="cursor-pointer text-left font-medium text-ink hover:text-teal-deep"
                            >
                              {r.name}
                            </button>
                            {r.guideline_flags.length > 0 && (
                              <span className="ml-2 inline-flex items-center gap-1 text-xs text-coral">
                                <TriangleAlert className="size-3.5" aria-hidden /> caution
                              </span>
                            )}
                          </td>
                          <td className="num py-2.5 pr-3 text-right">{pct(r.response_probability)}</td>
                          <td className="num py-2.5 pr-3 text-right">{r.expected_progression_free_months.toFixed(1)}</td>
                          <td className="py-2.5 pr-3">
                            <div className="flex items-center gap-2">
                              <div className="h-2 w-28 overflow-hidden rounded-full bg-line-soft">
                                <div className="h-full rounded-full bg-teal" style={{ width: `${(r.expected_survival_months / max) * 100}%` }} />
                              </div>
                              <span className="num">{r.expected_survival_months.toFixed(1)}</span>
                            </div>
                          </td>
                          <td className="num py-2.5 text-right">{pct(r.any_severe_side_effect)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {tab === "doctor" && (
          <div>
            <Heading as="h3">What the RL agent would choose</Heading>
            <p className="mt-1 text-sm text-ink-soft">
              A basic DQN agent trained only by treating simulated twins of training patients (Objective 3). It may not choose
              options excluded by the treatment constraints.
            </p>
            {!doctor && (
              <Button variant="primary" className="mt-4" busy={busy} onClick={() => run(() => api.aiDoctor(twinId), setDoctor)}>
                Ask the AI doctor
              </Button>
            )}
            {doctor && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-5 grid gap-5 md:grid-cols-[1fr_1.3fr]">
                <div className="self-start rounded-2xl border border-teal/40 bg-teal-soft/50 p-5">
                  <p className="text-xs uppercase tracking-[0.14em] text-teal-deep">Agent's choice</p>
                  <p className="mt-2 font-display text-2xl font-semibold text-ink">{doctor.choice}</p>
                  <Button className="mt-4" onClick={() => onPick(doctor.choice_key)}>See the twin's prediction for it</Button>
                </div>
                <div>
                  <p className="text-sm font-medium text-ink">Agent's top options (learned value of the whole 5-year journey)</p>
                  <ul className="mt-2 space-y-1.5">
                    {doctor.ranking.map((r) => {
                      const top = doctor.ranking[0].value;
                      const low = doctor.ranking[doctor.ranking.length - 1].value;
                      const width = 35 + 65 * ((r.value - low) / (top - low || 1));
                      return (
                        <li key={r.key} className="grid grid-cols-[1fr_8rem] items-center gap-3 text-sm">
                          <span className="text-ink-soft">{r.name}</span>
                          <div className="h-2 overflow-hidden rounded-full bg-line-soft">
                            <div className="h-full rounded-full bg-ink" style={{ width: `${width}%` }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                  {doctor.excluded_by_constraints.length > 0 && (
                    <p className="mt-3 text-xs text-muted">Excluded by constraints: {doctor.excluded_by_constraints.join(", ")}.</p>
                  )}
                  <p className="mt-3 text-xs leading-relaxed text-muted">{doctor.note}</p>
                </div>
              </motion.div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
