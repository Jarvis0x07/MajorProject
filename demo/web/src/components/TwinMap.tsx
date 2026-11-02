import { motion, AnimatePresence } from "framer-motion";
import type { Neighbor } from "../api";

interface Props {
  points: [number, number][];
  neighbors: Neighbor[] | null;
  position: [number, number] | null;
  phase: "idle" | "searching" | "built";
}

const W = 1000;
const H = 560;
const PAD = 28;
const sx = (x: number) => PAD + x * (W - 2 * PAD);
const sy = (y: number) => H - PAD - y * (H - 2 * PAD);

/** The twin database as a constellation: every stored twin is a dot; a new twin forms from its five nearest. */
export function TwinMap({ points, neighbors, position, phase }: Props) {
  const target = position ? { x: sx(position[0]), y: sy(position[1]) } : null;
  const near = neighbors?.map((n) => ({ ...n, x: sx(points[n.map_index][0]), y: sy(points[n.map_index][1]) })) ?? [];
  const summary = target
    ? `Map of ${points.length} stored twins. The new twin sits beside its ${near.length} most similar stored twins.`
    : `Map of ${points.length} stored twins, one per treatment decision of a training patient.`;

  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={summary}>
        <rect x="0" y="0" width={W} height={H} rx="18" fill="var(--color-paper)" />
        {points.map(([x, y], i) => (
          <circle key={i} cx={sx(x)} cy={sy(y)} r={2.6} fill="var(--color-ink)" opacity={phase === "built" ? 0.12 : 0.22} />
        ))}

        <AnimatePresence>
          {target && phase !== "idle" && (
            <motion.circle
              key="sweep"
              cx={target.x}
              cy={target.y}
              fill="none"
              stroke="var(--color-teal)"
              strokeWidth={1.5}
              initial={{ r: 0, opacity: 0.7 }}
              animate={{ r: 260, opacity: 0 }}
              transition={{ duration: 1.1, ease: "easeOut" }}
            />
          )}
        </AnimatePresence>

        {target && phase === "built" && (
          <g>
            {near.map((n, i) => (
              <motion.line
                key={`line-${n.rank}`}
                x1={n.x}
                y1={n.y}
                x2={target.x}
                y2={target.y}
                stroke="var(--color-teal)"
                strokeWidth={1.6}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.85 }}
                transition={{ delay: 0.15 + i * 0.08, duration: 0.45, ease: "easeOut" }}
              />
            ))}
            {near.map((n, i) => (
              <motion.g key={`n-${n.rank}`} initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.08, type: "spring", stiffness: 320, damping: 22 }} style={{ originX: `${n.x}px`, originY: `${n.y}px` }}>
                <circle cx={n.x} cy={n.y} r={9} fill="var(--color-teal-soft)" stroke="var(--color-teal)" strokeWidth={2} />
                <text x={n.x} y={n.y + 3.5} textAnchor="middle" className="num fill-teal-deep text-[10px] font-medium">{n.rank}</text>
              </motion.g>
            ))}
            <motion.g initial={{ opacity: 0, scale: 0.2 }} animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.65, type: "spring", stiffness: 260, damping: 18 }} style={{ originX: `${target.x}px`, originY: `${target.y}px` }}>
              <circle cx={target.x} cy={target.y} r={16} fill="var(--color-teal-deep)" opacity={0.15} />
              <path d={`M${target.x} ${target.y - 11} L${target.x + 11} ${target.y} L${target.x} ${target.y + 11} L${target.x - 11} ${target.y} Z`}
                fill="var(--color-teal-deep)" stroke="white" strokeWidth={2} />
            </motion.g>
          </g>
        )}
      </svg>
      <figcaption className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm text-ink-soft">
        <span className="inline-flex items-center gap-2"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="3" fill="var(--color-ink)" opacity="0.35" /></svg>Stored twin ({points.length.toLocaleString()} training decision points)</span>
        <span className="inline-flex items-center gap-2"><svg width="14" height="14" aria-hidden><circle cx="7" cy="7" r="5.5" fill="var(--color-teal-soft)" stroke="var(--color-teal)" strokeWidth="2" /></svg>5 nearest twins</span>
        <span className="inline-flex items-center gap-2"><svg width="14" height="14" aria-hidden><path d="M7 1 13 7 7 13 1 7Z" fill="var(--color-teal-deep)" /></svg>New twin for this patient</span>
        <span className="text-muted">Axes: first two principal components of the 37 patient features</span>
      </figcaption>
    </figure>
  );
}
