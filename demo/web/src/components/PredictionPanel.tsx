import { motion } from "framer-motion";
import type { Prediction } from "../api";
import { organName, pct } from "../format";
import { Caution, Card, Heading, Label, RiskBar, Stat } from "./primitives";

export function PredictionPanel({ prediction }: { prediction: Prediction }) {
  const p = prediction;
  return (
    <motion.div key={p.medication.key} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <Card>
        <Label>Step 4 · What the twin predicts</Label>
        <div className="mt-2">
          <Heading as="h2">{p.medication.name}</Heading>
        </div>
        <p className="mt-1 text-sm text-ink-soft">{p.medication.group} · predictions for this patient's twin only</p>

        {p.guideline_flags.length > 0 && (
          <div className="mt-4 space-y-2">
            {p.guideline_flags.map((flag) => (
              <Caution key={flag}>
                <span className="font-medium">Treatment constraint.</span> {flag} The twin's estimate below is shown as learned
                from past patients, who mostly received platinum when their outlook was better.
              </Caution>
            ))}
          </div>
        )}
        {!p.allowed_by_safety_rules && (
          <div className="mt-4">
            <Caution>This option breaks a safety rule for this patient (for example, earlier severe toxicity).</Caution>
          </div>
        )}

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <Stat label="Chance the tumour disappears" value={pct(p.response_probability)} detail={`± ${pct(p.uncertainty.response_std, 1)} across the twin's models`} />
          <Stat
            label="Months until it returns or grows"
            value={p.expected_progression_free_months.toFixed(1)}
            detail={`± ${p.uncertainty.progression_free_months_std.toFixed(1)} · average within 5 years`}
          />
          <Stat label="Alive after 1 year" value={pct(p.survival["12m"])} detail={`after 2 years ${pct(p.survival["24m"])}`} />
          <Stat
            label="Months alive (within 5 years)"
            value={p.expected_survival_months.toFixed(1)}
            detail={`± ${p.uncertainty.survival_months_std.toFixed(1)} across the twin's models`}
          />
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          {(["6m", "12m", "24m"] as const).map((k) => (
            <div key={k} className="rounded-lg border border-line-soft px-3 py-2">
              <p className="text-xs text-muted">Cancer under control at {k.replace("m", " months")}</p>
              <p className="num mt-1 text-lg text-ink">{pct(p.progression_free[k])}</p>
            </div>
          ))}
        </div>

        <div className="mt-6">
          <div className="flex items-baseline justify-between gap-4">
            <p className="text-sm font-medium text-ink">Risk of a severe side effect</p>
            <p className="num text-sm text-ink">any: {pct(p.side_effects.any_severe)}</p>
          </div>
          <p className="mt-1 text-xs text-muted">
            From drug-class side-effect profiles (assumptions: the dataset has no side-effect records), adjusted for age and past
            treatment.
          </p>
          <div className="mt-3 space-y-2">
            {p.side_effects.by_organ.map((o) => (
              <RiskBar key={o.organ} label={organName(o.organ)} probability={o.probability} level={o.level} />
            ))}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
