import { useRef, useState, type DragEvent } from "react";
import { motion } from "framer-motion";
import { Check, FileJson, Upload } from "lucide-react";
import { Card, Heading, Label } from "./primitives";

export const BUILD_STEPS = [
  "Read the patient record",
  "Encoded today's situation as 37 features",
  "Searched the stored twins",
  "Found the 5 most similar stored twins",
  "Built a new twin for this patient",
];

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  stepsDone: number;
  error: string | null;
  databaseSize: number | null;
}

export function UploadCard({ onFile, busy, stepsDone, error, databaseSize }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file && !busy) onFile(file);
  };

  return (
    <Card>
      <Label>Step 1 · Patient record</Label>
      <div className="mt-2">
        <Heading as="h2">Upload a record to build a twin</Heading>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">
        A JSON file in the <span className="num text-ink">ovarian-twin-record/v1</span> format: diagnosis, past treatments and
        check-ups up to today. Sample files are in <span className="num text-ink">demo/sample_patients</span>.
      </p>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a patient record JSON file"
        aria-disabled={busy}
        onClick={() => !busy && input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !busy && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`mt-5 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-8 text-center transition-colors duration-200 ${
          dragging ? "border-teal bg-teal-soft/60" : "border-line bg-paper/50 hover:border-teal/60"
        } ${busy ? "cursor-progress opacity-70" : ""}`}
      >
        <span className="grid size-12 place-items-center rounded-full bg-teal-soft text-teal-deep">
          <Upload className="size-5" aria-hidden />
        </span>
        <span className="text-sm font-medium text-ink">Drop a .json file here, or click to choose</span>
        <span className="text-xs text-muted">Read in your browser and sent only to this computer's twin server</span>
        <input
          ref={input}
          type="file"
          accept=".json,application/json"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <p role="alert" className="mt-4 rounded-xl border border-coral/30 bg-coral-soft px-4 py-3 text-sm text-ink">
          <span className="font-medium text-coral">Could not build a twin.</span> {error}
        </p>
      )}

      {(busy || stepsDone > 0) && (
        <ol className="mt-5 space-y-2" aria-live="polite">
          {BUILD_STEPS.map((step, i) => {
            const done = i < stepsDone;
            const label = i === 2 && databaseSize ? `Searched ${databaseSize.toLocaleString()} stored twins` : step;
            return (
              <motion.li
                key={step}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: done || i === stepsDone ? 1 : 0.4, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.03 }}
                className="flex items-center gap-3 text-sm"
              >
                <span className={`grid size-5 place-items-center rounded-full ${done ? "bg-teal text-white" : "border border-line text-muted"}`}>
                  {done ? <Check className="size-3.5" aria-hidden /> : <FileJson className="size-3" aria-hidden />}
                </span>
                <span className={done ? "text-ink" : "text-muted"}>{label}</span>
              </motion.li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}
