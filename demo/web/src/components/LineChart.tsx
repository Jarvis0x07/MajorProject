import { useId, useState, type MouseEvent } from "react";
import { motion } from "framer-motion";

export interface Series {
  label: string;
  values: number[];
  color: string;
  dashed?: boolean;
  markers?: boolean;
}

interface Props {
  x: number[];
  series: Series[];
  xLabel: string;
  yLabel: string;
  yMax?: number;
  yMin?: number;
  formatY?: (v: number) => string;
  formatX?: (v: number) => string;
  summary: string;
  height?: number;
}

const W = 640;
const PAD = { left: 52, right: 16, top: 16, bottom: 40 };

export function LineChart({ x, series, xLabel, yLabel, yMax = 1, yMin = 0, formatY = (v) => `${Math.round(v * 100)}%`, formatX = String, summary, height = 280 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useId();
  const plotW = W - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const xMin = x[0];
  const xMax = x[x.length - 1];
  const sx = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  const sy = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin || 1)) * plotH;
  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);
  const xTicks = x.length <= 8 ? x : x.filter((_, i) => i % Math.ceil(x.length / 7) === 0);

  const onMove = (event: MouseEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const value = xMin + ((event.clientX - box.left) / box.width) * (xMax - xMin);
    let best = 0;
    x.forEach((v, i) => {
      if (Math.abs(v - value) < Math.abs(x[best] - value)) best = i;
    });
    setHover(best);
  };

  return (
    <figure className="w-full">
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full" role="img" aria-label={summary}>
        <defs>
          <clipPath id={clipId}>
            <rect x={PAD.left} y={PAD.top - 4} width={plotW} height={plotH + 8} />
          </clipPath>
        </defs>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} stroke="var(--color-line-soft)" />
            <text x={PAD.left - 8} y={sy(t) + 4} textAnchor="end" className="num fill-muted text-[11px]">
              {formatY(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={height - PAD.bottom + 18} textAnchor="middle" className="num fill-muted text-[11px]">
            {formatX(t)}
          </text>
        ))}
        <text x={PAD.left + plotW / 2} y={height - 4} textAnchor="middle" className="fill-ink-soft text-[12px]">
          {xLabel}
        </text>
        <text transform={`translate(12 ${PAD.top + plotH / 2}) rotate(-90)`} textAnchor="middle" className="fill-ink-soft text-[12px]">
          {yLabel}
        </text>
        <g clipPath={`url(#${clipId})`}>
          {series.map((s) => {
            const d = s.values.map((v, i) => `${i ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
            return (
              <g key={s.label}>
                {/* Drawing animations rewrite the dash pattern, so dashed series fade in instead. */}
                <motion.path
                  d={d}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2.5}
                  strokeDasharray={s.dashed ? "6 5" : undefined}
                  strokeLinejoin="round"
                  initial={s.dashed ? { opacity: 0 } : { pathLength: 0 }}
                  animate={s.dashed ? { opacity: 1 } : { pathLength: 1 }}
                  transition={{ duration: 0.9, ease: "easeOut" }}
                />
                {s.markers && s.values.map((v, i) => <circle key={i} cx={sx(x[i])} cy={sy(v)} r={3.5} fill={s.color} />)}
              </g>
            );
          })}
        </g>
        {hover !== null && (
          <g pointerEvents="none">
            <line x1={sx(x[hover])} x2={sx(x[hover])} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--color-ink-soft)" strokeDasharray="2 3" />
            {series.map((s) => (
              <circle key={s.label} cx={sx(x[hover])} cy={sy(s.values[hover])} r={4.5} fill="white" stroke={s.color} strokeWidth={2} />
            ))}
          </g>
        )}
        <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </svg>
      <figcaption className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-ink-soft">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-2">
            <svg width="22" height="8" aria-hidden>
              <line x1="0" x2="22" y1="4" y2="4" stroke={s.color} strokeWidth="2.5" strokeDasharray={s.dashed ? "5 4" : undefined} />
            </svg>
            {s.label}
            {hover !== null && <span className="num text-ink">{formatY(s.values[hover])}</span>}
          </span>
        ))}
        {hover !== null && <span className="num text-muted">at {formatX(x[hover])}</span>}
      </figcaption>
    </figure>
  );
}
