import { useRef, useState, type DragEvent } from "react";
import { Upload } from "lucide-react";
import { Card, Heading, Label } from "./primitives";

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  error: string | null;
}

export function UploadCard({ onFile, busy, error }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const open = () => !busy && input.current?.click();

  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file && !busy) onFile(file);
  };

  return (
    <Card>
      <Label>Step 1 · Patient record</Label>
      <div className="mt-2">
        <Heading>Upload a patient</Heading>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">
        A JSON file with the diagnosis, past treatments and check-ups up to today. Samples are in{" "}
        <span className="num text-ink">demo/sample_patients</span>.
      </p>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a patient record JSON file"
        aria-disabled={busy}
        onClick={open}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && open()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
        className={`mt-5 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-8 text-center transition-colors duration-200 ${
          dragging ? "border-teal bg-teal-soft/60" : "border-line bg-paper/50 hover:border-teal/60"
        } ${busy ? "cursor-progress opacity-70" : ""}`}
      >
        <span className="grid size-12 place-items-center rounded-full bg-teal-soft text-teal-deep">
          {busy ? (
            <span className="size-5 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />
          ) : (
            <Upload className="size-5" aria-hidden />
          )}
        </span>
        <span className="text-sm font-medium text-ink">{busy ? "Building the twin…" : "Drop a .json file here, or click to choose"}</span>
        <span className="text-xs text-muted">Sent only to the twin server running on this computer</span>
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
    </Card>
  );
}
