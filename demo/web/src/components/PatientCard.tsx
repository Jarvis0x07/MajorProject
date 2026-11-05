import { motion } from "framer-motion";
import type { Analysis } from "../api";
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

const change = (multiplier: number) => {
  const percent = Math.round((multiplier - 1) * 100);
  return percent === 0 ? "unchanged" : `${percent > 0 ? "+" : ""}${percent}%`;
};

export function PatientCard({ analysis }: { analysis: Analysis }) {
  const p = analysis.patient;
  return (
    <Card>
      <Label>Step 2 · The patient's digital twin</Label>
      <div className="mt-2">
        <Heading>{analysis.label}</Heading>
      </div>
      <p className="mt-1 text-sm text-ink-soft">
        Today: <span className="num text-ink">{analysis.today_month.toFixed(1)}</span> months after diagnosis · deciding treatment line{" "}
        <span className="num text-ink">{p.treatment_line}</span>
      </p>

      <dl className="mt-4 grid grid-cols-2 gap-2">
        <Fact label="Age now" value={p.age === null ? "Not recorded" : `${p.age} years`} />
        <Fact label="Situation" value={p.situation} />
        <Fact label="At diagnosis" value={`${stageName(p.stage)} · ${gradeName(p.grade)}`} />
        <Fact label="Tumour left after surgery" value={p.residual} />
        <Fact
          label="Last platinum"
          value={p.months_since_platinum === null ? "Never given" : `${p.months_since_platinum} months ago`}
        />
        <Fact label="Platinum status" value={p.months_since_platinum === null ? "—" : p.platinum_resistant ? "Resistant (< 6 months)" : "Sensitive"} />
      </dl>

      {p.past_treatments.length > 0 && (
        <div className="mt-5">
          <p className="text-sm font-medium text-ink">Past treatment lines</p>
          <ol className="mt-2 space-y-1.5">
            {p.past_treatments.map((t, i) => (
              <li key={i} className="flex items-baseline gap-3 text-sm">
                <span className="num w-20 shrink-0 text-xs text-muted">month {t.start_month}</span>
                <span className="text-ink">{t.drugs.join(" + ")}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="mt-6 border-t border-line-soft pt-5">
        <p className="text-sm font-medium text-ink">Built from the 5 most similar training patients</p>
        <p className="mt-1 text-xs leading-relaxed text-muted">
          Matched on age, stage, grade, residual tumour, treatment line, situation and past drugs. How these patients really did,
          compared with what the model expected for them, adjusts this twin's risks: progression{" "}
          <span className="num text-ink">{change(analysis.risk_multipliers.progression)}</span>, death{" "}
          <span className="num text-ink">{change(analysis.risk_multipliers.death)}</span>.
        </p>
        <ul className="mt-3 space-y-2">
          {analysis.neighbours.map((n, i) => (
            <motion.li
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.06, duration: 0.25 }}
              className="grid grid-cols-[1.5rem_1fr_3rem] items-center gap-3 text-sm"
            >
              <span className="num grid size-6 place-items-center rounded-full bg-teal-soft text-xs text-teal-deep">{i + 1}</span>
              <span className="text-ink-soft">
                {n.age ?? "?"} y · {stageName(n.stage)} · line {n.treatment_line} · {n.situation.toLowerCase()}
                {n.past_drugs.length > 0 && <span className="text-muted"> · had {n.past_drugs.join(", ").toLowerCase()}</span>}
              </span>
              <span className="num text-right text-ink">{Math.round(n.similarity * 100)}%</span>
            </motion.li>
          ))}
        </ul>
      </div>
    </Card>
  );
}
