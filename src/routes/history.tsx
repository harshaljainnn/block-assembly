import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Badge, Card, Page, inputCls, td, th } from "@/components/assembly/ui";
import { historyRows, type Cycle } from "@/lib/assembly-data";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Assembly History — Assembly Monitor" },
      { name: "description", content: "Review completed assembly cycles, durations and failure reasons." },
      { property: "og:title", content: "Assembly History — Assembly Monitor" },
      { property: "og:description", content: "Review completed assembly cycles, durations and failure reasons." },
    ],
  }),
  component: History,
});

const tone = (s: Cycle["status"]) => (s === "Failed" ? "danger" : s === "In Progress" ? "warning" : "success");

function History() {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [open, setOpen] = useState<Cycle | null>(null);
  const rows = historyRows.filter((r) => (!q || r.id.toLowerCase().includes(q.toLowerCase())) && (!status || r.status === status));

  return (
    <Page title="Assembly History" subtitle="Review completed assembly cycles.">
      <div className="my-[18px] flex flex-wrap gap-2.5">
        <input className={inputCls} placeholder="Search Cycle ID" value={q} onChange={(e) => setQ(e.target.value)} />
        <select className={inputCls} value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">All statuses</option><option>Complete</option><option>Failed</option><option>In Progress</option>
        </select>
        <select className={inputCls}><option>Today</option><option>7 Days</option><option>30 Days</option></select>
      </div>
      <Card>
        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead><tr>{["Cycle ID", "Start Time", "End Time", "Duration", "States", "Status", "Failure Reason"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
            <tbody>
              {rows.length ? rows.map((r) => (
                <tr key={r.id} onClick={() => setOpen(r)} className={cn("cursor-pointer", r.status === "Failed" && "bg-destructive-soft")}>
                  <td className={td}><b>{r.id}</b></td><td className={td}>{r.start_time}</td><td className={td}>{r.end_time}</td>
                  <td className={td}>{r.duration}s</td><td className={td}>{r.state}</td>
                  <td className={td}><Badge tone={tone(r.status)}>{r.status}</Badge></td>
                  <td className={cn(td, r.status === "Failed" && "text-destructive")}>{r.failure_reason || "—"}</td>
                </tr>
              )) : (
                <tr><td colSpan={7}><div className="p-10 text-center text-sm text-muted-foreground">No assembly cycles found.</div></td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {open && (
        <div className="fixed inset-0 bg-foreground/20" onClick={() => setOpen(null)}>
          <div className="absolute right-0 top-0 h-full w-[min(520px,92vw)] overflow-auto bg-card p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Cycle Details</h2>
              <button className="cursor-pointer text-2xl" onClick={() => setOpen(null)} aria-label="Close">×</button>
            </div>
            {[
              ["Cycle ID", open.id], ["Duration", `${open.duration}s`], ["Status", open.status],
              ["Failure Reason", open.failure_reason || "—"], ["State Sequence", open.state],
              ["Events", "Event details will populate from recorded assembly events."],
            ].map(([k, v]) => (
              <div key={k} className="border-b border-border py-3">
                <div className="text-[11px] text-muted-foreground">{k}</div>
                <div className={cn("mt-1 font-semibold", k === "Failure Reason" && open.status === "Failed" && "text-destructive")}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Page>
  );
}
