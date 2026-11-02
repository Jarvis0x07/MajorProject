import { motion } from "framer-motion";
import type { TwinSummary } from "../api";
import { gradeName, stageName } from "../format";
import { Card, Heading, Label } from "./primitives";

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-paper/70 px-3 py-2">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-ink">{value}</dd>
    </div>
  );
}

export function PatientCard({ twin }: { twin: TwinSummary }) {
  const p = twin.patient;
  return (
    <Card>
      <Label>Step 2 · The new twin</Label>
      <div className="mt-2 flex items-baseline justify-between gap-4">
        <Heading as="h2">{twin.label}</Heading>
        <span className="num text-xs text-muted">twin {twin.id}</span>
      </div>
      <p className="mt-1 text-sm text-ink-soft">
        Today: <span className="num text-ink">{twin.today_month.toFixed(1)}</span> months after diagnosis · treatment line{" "}
        <span className="num text-ink">{p.treatment_line}</span>
      </p>

      <dl className="mt-4 grid grid-cols-2 gap-2">
        <Fact label="Age now" value={p.age_now ? `${p.age_now.toFixed(0)} years` : "Not recorded"} />
        <Fact label="Situation" value={p.situation} />
        <Fact label="Cancer at diagnosis" value={`${stageName(p.stage)} · ${gradeName(p.grade)}`} />
        <Fact label="Tumour left after surgery" value={p.residual ?? "Not recorded"} />
        <Fact label="Last check-up" value={p.last_tumor_status} />
        <Fact label="Platinum" value={p.months_since_last_platinum === null ? "Not given yet" : `${p.months_since_last_platinum} months since last`} />
      </dl>
      <p className="mt-3 text-sm text-ink-soft">{p.platinum_status}</p>

      {p.past_treatments.length > 0 && (
        <div className="mt-5">
          <p className="text-sm font-medium text-ink">Past treatment lines</p>
          <ol className="mt-2 space-y-1.5">
            {p.past_treatments.map((t) => (
              <li key={t.line} className="flex items-baseline gap-3 text-sm">
                <span className="num w-24 shrink-0 whitespace-nowrap text-xs text-muted">month {t.start_month}</span>
                <span className="text-ink">{t.classes.join(" + ")}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="mt-6 border-t border-line-soft pt-5">
        <p className="text-sm font-medium text-ink">Built from the 5 most similar stored twins</p>
        <p className="mt-1 text-xs leading-relaxed text-muted">
          Similar by age, stage, grade, residual tumour, treatment line, situation and past drugs. Their outcomes are not shown:
          they only tune how this twin's progression risk is calibrated (×{twin.build.progression_risk_multiplier.toFixed(2)}).
        </p>
        <ul className="mt-3 space-y-2">
          {twin.build.neighbors.map((n, i) => (
            <motion.li
              key={n.rank}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.05, duration: 0.25 }}
              className="grid grid-cols-[1.5rem_1fr_3.5rem] items-center gap-3 text-sm"
            >
              <span className="num grid size-6 place-items-center rounded-full bg-teal-soft text-xs text-teal-deep">{n.rank}</span>
              <span className="text-ink-soft">
                {n.age ?? "?"} y · {stageName(n.stage)} · line {n.treatment_line} · {n.situation.toLowerCase()}
                {n.prior_classes.length > 0 && <span className="text-muted"> · had {n.prior_classes.join(", ").toLowerCase()}</span>}
              </span>
              <span className="num text-right text-ink">{n.similarity}%</span>
            </motion.li>
          ))}
        </ul>
      </div>
    </Card>
  );
}
