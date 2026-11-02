import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-line bg-surface/90 p-6 shadow-card backdrop-blur-sm ${className}`}>
      {children}
    </section>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <p className="text-xs font-medium uppercase tracking-[0.14em] text-muted">{children}</p>;
}

export function Heading({ children, as: Tag = "h2" }: { children: ReactNode; as?: "h1" | "h2" | "h3" }) {
  const size = Tag === "h1" ? "text-4xl md:text-5xl" : Tag === "h2" ? "text-2xl" : "text-lg";
  return <Tag className={`font-display ${size} font-semibold leading-tight tracking-tight text-ink`}>{children}</Tag>;
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary"; busy?: boolean };

export function Button({ variant = "secondary", busy = false, children, className = "", disabled, ...rest }: ButtonProps) {
  const styles =
    variant === "primary"
      ? "bg-teal text-white hover:bg-teal-deep border-teal"
      : "bg-surface text-ink hover:border-teal hover:text-teal-deep border-line";
  return (
    <button
      {...rest}
      disabled={disabled || busy}
      aria-busy={busy}
      className={`inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-xl border px-4 text-sm font-medium transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
    >
      {busy && <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />}
      {children}
    </button>
  );
}

export function Stat({ label, value, detail }: { label: string; value: ReactNode; detail?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line-soft bg-paper/60 p-4">
      <p className="text-sm text-ink-soft">{label}</p>
      <p className="num mt-1 text-3xl font-medium text-ink">{value}</p>
      {detail && <p className="mt-1 text-xs text-muted">{detail}</p>}
    </div>
  );
}

const LEVEL_STYLE: Record<string, string> = {
  high: "bg-coral",
  moderate: "bg-amber",
  low: "bg-teal",
  none: "bg-line",
};

export function RiskBar({ label, probability, level }: { label: string; probability: number; level: string }) {
  return (
    <div className="grid grid-cols-[7rem_1fr_6.5rem] items-center gap-3 text-sm sm:grid-cols-[10rem_1fr_7rem]">
      <span className="text-ink-soft">{label}</span>
      <div className="h-2.5 overflow-hidden rounded-full bg-line-soft" role="presentation">
        <div className={`h-full rounded-full ${LEVEL_STYLE[level] ?? "bg-teal"}`} style={{ width: `${Math.min(100, probability * 200)}%` }} />
      </div>
      <span className="num whitespace-nowrap text-right text-ink">
        {(probability * 100).toFixed(0)}% <span className="text-xs text-muted">{level}</span>
      </span>
    </div>
  );
}

export function Caution({ children }: { children: ReactNode }) {
  return (
    <div role="note" className="flex gap-3 rounded-xl border border-coral/30 bg-coral-soft px-4 py-3 text-sm text-ink">
      <svg viewBox="0 0 24 24" className="mt-0.5 size-5 shrink-0 text-coral" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
        <path d="M12 3 2 21h20L12 3Z" strokeLinejoin="round" />
        <path d="M12 10v5M12 18h.01" strokeLinecap="round" />
      </svg>
      <div>{children}</div>
    </div>
  );
}
