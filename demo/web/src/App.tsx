import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { analyse, type Analysis } from "./api";
import { EffectivenessTable } from "./components/EffectivenessTable";
import { PatientCard } from "./components/PatientCard";
import { Card, Heading, Label } from "./components/primitives";
import { UploadCard } from "./components/UploadCard";

const STEPS = [
  ["Upload", "A patient record: diagnosis, past treatments and check-ups up to today."],
  ["Twin", "The outcome model at this patient's state, adjusted by how the 5 most similar training patients really did."],
  ["Effectiveness", "The twin predicts response and survival for every medication; the RL agent picks one."],
];

export default function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const results = useRef<HTMLDivElement>(null);

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await analyse(file));
      requestAnimationFrame(() => {
        if (window.innerWidth < 1024) results.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-paper/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-8">
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-teal-deep">Digital twins · Reinforcement learning</p>
          <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            Ovarian Twin <span className="italic text-teal">Lab</span>
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-soft">
            Upload a patient, build their digital twin from the 5 most similar past patients, and see how effective each medication
            would be.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-8 sm:px-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,8fr)]">
        <div className="space-y-6">
          <UploadCard onFile={upload} busy={busy} error={error} />
          <AnimatePresence mode="wait">
            {analysis && (
              <motion.div key={analysis.label + analysis.today_month} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <PatientCard analysis={analysis} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div ref={results} className="scroll-mt-6">
          <AnimatePresence mode="wait">
            {analysis ? (
              <motion.div key={analysis.label + analysis.today_month} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <EffectivenessTable analysis={analysis} />
              </motion.div>
            ) : (
              <motion.div key="intro" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card>
                  <Label>How it works</Label>
                  <div className="mt-2">
                    <Heading>From one record to a ranked list of treatments</Heading>
                  </div>
                  <ol className="mt-5 space-y-3">
                    {STEPS.map(([title, text], i) => (
                      <motion.li
                        key={title}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 + i * 0.08 }}
                        className="flex gap-4 rounded-xl border border-line-soft bg-paper/60 p-4"
                      >
                        <span className="num grid size-8 shrink-0 place-items-center rounded-full bg-teal text-sm text-white">{i + 1}</span>
                        <div>
                          <p className="font-medium text-ink">{title}</p>
                          <p className="mt-0.5 text-sm text-ink-soft">{text}</p>
                        </div>
                      </motion.li>
                    ))}
                  </ol>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <footer className="border-t border-line">
        <p className="mx-auto max-w-7xl px-4 py-6 text-xs leading-relaxed text-muted sm:px-8">
          Research prototype on the public, de-identified TCGA-OV clinical dataset. Predictions are associations learned from
          historical records, not proof that a treatment works. Not medical advice.
        </p>
      </footer>
    </div>
  );
}
