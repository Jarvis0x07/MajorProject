import { TriangleAlert } from "lucide-react";
import type { ComparisonRow, Medication } from "../api";
import { Card, Heading, Label } from "./primitives";

interface Props {
  medications: Medication[];
  comparison: ComparisonRow[] | null;
  selected: string | null;
  onSelect: (key: string) => void;
  disabled: boolean;
}

export function MedicationPicker({ medications, comparison, selected, onSelect, disabled }: Props) {
  const flagged = new Set(comparison?.filter((r) => r.guideline_flags.length > 0).map((r) => r.key));
  return (
    <Card>
      <Label>Step 3 · Choose a medication</Label>
      <div className="mt-2">
        <Heading as="h2">What if we give…</Heading>
      </div>
      <p className="mt-2 text-sm text-ink-soft">
        The 14 regimens the twin learned from real treatment lines. The twin reasons about drug classes, so names are typical
        examples of each class.
      </p>
      <div className="mt-5 grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Medication">
        {medications.map((m) => {
          const active = m.key === selected;
          return (
            <button
              key={m.key}
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onSelect(m.key)}
              className={`group flex min-h-16 cursor-pointer flex-col items-start justify-center rounded-xl border px-4 py-3 text-left transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50 ${
                active ? "border-teal bg-teal-soft/70" : "border-line bg-surface hover:border-teal/60"
              }`}
            >
              <span className={`text-sm font-medium ${active ? "text-teal-deep" : "text-ink"}`}>{m.name}</span>
              <span className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
                {m.group}
                {flagged.has(m.key) && (
                  <span className="inline-flex items-center gap-1 text-coral">
                    <TriangleAlert className="size-3.5" aria-hidden /> guideline caution
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}
