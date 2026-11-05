import { motion } from "framer-motion";
import { Bot, TriangleAlert, Trophy } from "lucide-react";
import type { Analysis, Medication } from "../api";
import { pct } from "../format";
import { Card, Caution, Heading, Label } from "./primitives";

const HORIZON_MONTHS = 60;

function Highlight({ icon: Icon, title, medication, note }: { icon: typeof Bot; title: string; medication: Medication; note: string }) {
  return (
    <div className="rounded-2xl border border-teal/30 bg-gradient-to-br from-teal-soft/80 to-surface p-5">
      <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-teal-deep">
        <Icon className="size-4" aria-hidden /> {title}
      </p>
      <p className="mt-2 font-display text-xl font-semibold leading-snug text-ink">{medication.name}</p>
      <p className="mt-1 text-sm text-ink-soft">
        <span className="num text-ink">{medication.survival_months.toFixed(1)}</span> months alive expected within 5 years
      </p>
      <p className="mt-2 text-xs leading-relaxed text-muted">{note}</p>
    </div>
  );
}

export function EffectivenessTable({ analysis }: { analysis: Analysis }) {
  const meds = analysis.medications;
  const best = meds.find((m) => !m.guideline_caution) ?? meds[0];
  const agent = meds.find((m) => m.key === analysis.agent_choice) ?? best;

  return (
    <Card>
      <Label>Step 3 · Effectiveness of each medication</Label>
      <div className="mt-2">
        <Heading>What each medication would do for this twin</Heading>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">
        Every regimen the model learned from real treatment lines, given to the twin today. Ranked by expected months alive within
        5 years. The twin works on drug classes, so drug names are typical examples.
      </p>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <Highlight icon={Trophy} title="Twin's best option" medication={best} note="Longest expected survival for this line, among the options guidelines allow." />
        <Highlight
          icon={Bot}
          title="RL agent's choice"
          medication={agent}
          note="The DQN agent plans the whole 5-year treatment journey it learned by treating twins, so it can differ from the single-line ranking."
        />
      </div>

      {analysis.patient.platinum_resistant && (
        <div className="mt-4">
          <Caution>
            <span className="font-medium">Platinum-resistant.</span> The cancer came back within 6 months of platinum, and guidelines
            advise against giving platinum again. Platinum options are marked; the twin's numbers for them are learned from past
            patients, who mostly received platinum when their outlook was better.
          </Caution>
        </div>
      )}

      <div className="mt-5 -mx-2 overflow-x-auto px-2">
        <table className="w-full min-w-[42rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-muted">
              <th className="py-2 pr-3 font-medium">Medication</th>
              <th className="py-2 pr-3 text-right font-medium">Tumour disappears</th>
              <th className="py-2 pr-3 text-right font-medium">Months until it returns</th>
              <th className="py-2 pr-3 text-right font-medium">Alive at 1 year</th>
              <th className="py-2 font-medium">Months alive within 5 years</th>
            </tr>
          </thead>
          <tbody>
            {meds.map((m, i) => (
              <motion.tr
                key={m.key}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 + i * 0.03, duration: 0.25 }}
                className={`border-b border-line-soft transition-colors hover:bg-teal-soft/30 ${m.guideline_caution ? "text-ink-soft" : ""}`}
              >
                <td className="py-3 pr-3">
                  <div className="flex items-baseline gap-2">
                    <span className="num w-5 shrink-0 text-xs text-muted">{i + 1}</span>
                    <div>
                      <p className="font-medium text-ink">{m.name}</p>
                      <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
                        {m.group}
                        {m.key === analysis.agent_choice && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-teal-soft px-2 py-0.5 text-teal-deep">
                            <Bot className="size-3" aria-hidden /> RL agent
                          </span>
                        )}
                        {m.guideline_caution && (
                          <span className="inline-flex items-center gap-1 text-coral">
                            <TriangleAlert className="size-3.5" aria-hidden /> guideline caution
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                </td>
                <td className="num py-3 pr-3 text-right">{pct(m.response)}</td>
                <td className="num py-3 pr-3 text-right">{m.progression_free_months.toFixed(1)}</td>
                <td className="num py-3 pr-3 text-right">{pct(m.survival_12m)}</td>
                <td className="py-3">
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-28 overflow-hidden rounded-full bg-line-soft" role="presentation">
                      <motion.div
                        className={`h-full rounded-full ${m.guideline_caution ? "bg-coral/60" : "bg-gradient-to-r from-teal to-teal-deep"}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${(m.survival_months / HORIZON_MONTHS) * 100}%` }}
                        transition={{ delay: 0.15 + i * 0.03, duration: 0.5, ease: "easeOut" }}
                      />
                    </div>
                    <span className="num whitespace-nowrap">
                      {m.survival_months.toFixed(1)} <span className="text-xs text-muted">± {m.survival_months_std.toFixed(1)}</span>
                    </span>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-muted">
        ± is the spread between the twin's 10 networks: larger means the model is less sure, usually because few past patients
        received that option in a situation like this one.
      </p>
    </Card>
  );
}
