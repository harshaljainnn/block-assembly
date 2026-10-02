import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function TopBar() {
  const links = [
    { to: "/", label: "Live Monitor" },
    { to: "/operators", label: "Operators" },
    { to: "/history", label: "History" },
    { to: "/analytics", label: "Analytics" },
  ] as const;

  return (
    <header className="h-16 border-b border-border bg-card flex items-center">
      <div className="mx-auto flex w-full max-w-[1200px] items-center justify-between px-4 md:px-6">
        <Link to="/" className="font-bold">
          Assembly Monitor
        </Link>
        <nav className="flex items-center gap-2.5 text-xs sm:gap-3 sm:text-sm md:gap-[22px]">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              activeOptions={{ exact: true }}
              className="text-muted-foreground hover:text-foreground transition-colors"
              activeProps={{ className: "!text-primary font-semibold" }}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

export function Page({
  title,
  subtitle,
  right,
  children,
}: {
  title: string;
  subtitle: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      <TopBar />
      <main className="mx-auto max-w-[1200px] px-4 py-6 md:px-6 md:py-8">
        <div className="mb-6 flex flex-col items-start justify-between gap-3 sm:flex-row">
          <div>
            <h1 className="text-[21px] font-bold md:text-2xl">{title}</h1>
            <p className="mt-1.5 text-sm text-muted-foreground">{subtitle}</p>
          </div>
          {right}
        </div>
        {children}
      </main>
    </div>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("rounded-[10px] border border-border bg-card p-5 shadow-sm", className)}>
      {children}
    </div>
  );
}

const accents = ["border-t-primary", "border-t-success", "border-t-amber", "border-t-sky"];

export function KpiGrid({
  items,
  className,
}: {
  items: { label: string; value: ReactNode; note?: ReactNode }[];
  className?: string;
}) {
  return (
    <section className={cn("grid grid-cols-2 gap-3 md:grid-cols-4", className)}>
      {items.map((k, i) => (
        <Card key={k.label} className={cn("border-t-[3px]", accents[i % 4])}>
          <div className="text-xs text-muted-foreground">{k.label}</div>
          <div className="mt-2 text-2xl font-bold">{k.value}</div>
          {k.note && <div className="mt-1 text-[11px] text-muted-foreground">{k.note}</div>}
        </Card>
      ))}
    </section>
  );
}

export function Badge({
  tone,
  children,
}: {
  tone: "success" | "danger" | "warning";
  children: ReactNode;
}) {
  const t = {
    success: "bg-success-soft text-success",
    danger: "bg-destructive-soft text-destructive",
    warning: "bg-warning-soft text-warning",
  }[tone];
  return (
    <span
      className={cn("inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-bold", t)}
    >
      {children}
    </span>
  );
}

export function Progress({
  value,
  className,
  barClass,
}: {
  value: number;
  className?: string;
  barClass?: string;
}) {
  return (
    <div className={cn("h-2 overflow-hidden rounded-full bg-muted", className)}>
      <span
        className={cn("block h-full bg-primary", barClass)}
        style={{ width: `${Math.min(100, value)}%` }}
      />
    </div>
  );
}

export function Btn({ className, ...p }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...p}
      className={cn(
        "cursor-pointer rounded-lg border border-border bg-card px-3 py-2 text-xs hover:bg-accent hover:text-primary",
        className,
      )}
    />
  );
}

export const inputCls =
  "rounded-lg border border-border bg-card px-3 py-2 text-[13px] min-w-40 focus:outline-2 focus:outline-primary/30 focus:border-primary";
export const th =
  "whitespace-nowrap border-b border-border p-3 text-left text-[11px] font-semibold text-muted-foreground";
export const td = "border-b border-border px-3 py-3 align-top";
