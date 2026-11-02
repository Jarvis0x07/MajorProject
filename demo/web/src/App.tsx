import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { api, type ComparisonRow, type Prediction, type ProjectInfo, type TwinSummary } from "./api";
import { ExplorePanel } from "./components/ExplorePanel";
import { MedicationPicker } from "./components/MedicationPicker";
import { MethodPage } from "./components/MethodPage";
import { PatientCard } from "./components/PatientCard";
import { PredictionPanel } from "./components/PredictionPanel";
import { Card, Heading, Label } from "./components/primitives";
import { TwinMap } from "./components/TwinMap";
import { BUILD_STEPS, UploadCard } from "./components/UploadCard";

type View = "lab" | "method";
const STEP_MS = 280;
const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function App() {
  const [view, setView] = useState<View>("lab");
  const [info, setInfo] = useState<ProjectInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [twin, setTwin] = useState<TwinSummary | null>(null);
  const [phase, setPhase] = useState<"idle" | "searching" | "built">("idle");
  const [stepsDone, setStepsDone] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonRow[] | null>(null);
  const [medication, setMedication] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.info().then(setInfo).catch((e) => setInfoError(e instanceof Error ? e.message : String(e)));
  }, []);

  const buildTwin = async (file: File) => {
    setUploadError(null);
    setTwin(null);
    setComparison(null);
    setMedication(null);
    setPrediction(null);
    setStepsDone(0);
    setPhase("searching");
    try {
      // Each step really happens on the server; the pacing only makes the sequence readable on screen.
      const request = api.createTwin(file);
      for (let i = 1; i <= 3; i++) {
        await wait(STEP_MS);
        setStepsDone(i);
      }
      const built = await request;
      setTwin(built);
      setStepsDone(4);
      setPhase("built");
      await wait(STEP_MS * 2);
      setStepsDone(BUILD_STEPS.length);
      setComparison(await api.compare(built.id));
    } catch (e) {
      setPhase("idle");
      setStepsDone(0);
      setUploadError(e instanceof Error ? e.message : String(e));
    }
  };

  const choose = async (key: string) => {
    if (!twin) return;
    setMedication(key);
    setPredictError(null);
    try {
      setPrediction(await api.predict(twin.id, key));
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      setPredictError(e instanceof Error ? e.message : String(e));
    }
  };

  const medicationName = info?.medications.find((m) => m.key === medication)?.name ?? "";

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:rounded-lg focus:bg-surface focus:px-4 focus:py-2">
        Skip to content
      </a>
      <header className="border-b border-line bg-paper/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-6 px-4 py-6 sm:px-8">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-teal-deep">Major Project · Objectives 1–3</p>
            <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
              Ovarian Twin <span className="italic text-teal">Lab</span>
            </h1>
            <p className="mt-1 max-w-xl text-sm text-ink-soft">
              Build a patient's digital twin from their record, then test treatments on the twin instead of the patient.
            </p>
          </div>
          <nav aria-label="Sections" className="flex gap-1 rounded-xl border border-line bg-surface p-1">
            {(["lab", "method"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                aria-current={view === v ? "page" : undefined}
                className={`min-h-10 cursor-pointer rounded-lg px-4 text-sm font-medium transition-colors duration-200 ${
                  view === v ? "bg-ink text-paper" : "text-ink-soft hover:text-ink"
                }`}
              >
                {v === "lab" ? "Twin Lab" : "How it works"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-7xl px-4 py-8 sm:px-8">
        {infoError && (
          <Card className="mb-6 border-coral/40">
            <p role="alert" className="text-sm text-ink">
              <span className="font-medium text-coral">The twin server is not reachable.</span> Start it with{" "}
              <span className="num">python -m demo</span> from the project folder. ({infoError})
            </p>
          </Card>
        )}

        <AnimatePresence mode="wait">
          {view === "method" && info ? (
            <motion.div key="method" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              <MethodPage info={info} />
            </motion.div>
          ) : (
            <motion.div key="lab" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}
              className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
              <div className="space-y-6">
                <UploadCard onFile={buildTwin} busy={phase === "searching"} stepsDone={stepsDone} error={uploadError}
                  databaseSize={info?.cohort.stored_twins ?? null} />
                {twin && <PatientCard twin={twin} />}
              </div>

              <div className="space-y-6">
                <Card>
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <Label>The twin database</Label>
                      <div className="mt-2">
                        <Heading as="h2">{twin ? "A new twin, formed from its five nearest" : "Every stored twin, mapped by similarity"}</Heading>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4">
                    {info ? (
                      <TwinMap points={info.twin_map} neighbors={twin?.build.neighbors ?? null} position={twin?.map_position ?? null} phase={phase} />
                    ) : (
                      <div className="aspect-[1000/560] animate-pulse rounded-2xl bg-line-soft" aria-label="Loading map" />
                    )}
                  </div>
                </Card>

                {twin && info && (
                  <MedicationPicker medications={info.medications} comparison={comparison} selected={medication} onSelect={choose}
                    disabled={phase !== "built"} />
                )}
                {predictError && <p role="alert" className="text-sm text-coral">{predictError}</p>}
                <div ref={resultRef} className="scroll-mt-6 space-y-6">
                  {prediction && <PredictionPanel prediction={prediction} />}
                  {twin && medication && prediction && (
                    <ExplorePanel twinId={twin.id} medication={medication} medicationName={medicationName} comparison={comparison} onPick={choose} />
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="border-t border-line">
        <p className="mx-auto max-w-7xl px-4 py-6 text-xs leading-relaxed text-muted sm:px-8">
          Research prototype built on the public, de-identified TCGA-OV clinical dataset. Predictions are associations learned
          from historical records, and side-effect risks are assumptions. Not medical advice.
        </p>
      </footer>
    </div>
  );
}
