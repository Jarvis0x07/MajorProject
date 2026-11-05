import type { ReactNode } from "react";

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

export function Heading({ children }: { children: ReactNode }) {
  return <h2 className="font-display text-2xl font-semibold leading-tight tracking-tight text-ink">{children}</h2>;
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
