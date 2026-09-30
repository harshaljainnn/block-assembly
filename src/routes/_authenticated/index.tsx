import { createFileRoute } from "@tanstack/react-router";
import { Badge, Btn, Card, KpiGrid, Page, Progress, td, th } from "@/components/assembly/ui";
import { liveEvents } from "@/lib/assembly-data";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_authenticated/")({
  head: () => ({
    meta: [
      { title: "Live Monitor — Assembly Monitor" },
      { name: "description", content: "Live view of the current assembly cycle, camera feed and latest events." },
      { property: "og:title", content: "Live Monitor — Assembly Monitor" },
      { property: "og:description", content: "Live view of the current assembly cycle, camera feed and latest events." },
    ],
  }),
  component: LiveMonitor,
});

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
    </div>
  );
}

function LiveMonitor() {
  return (
    <Page
      title="Live Monitor"
      subtitle="Current assembly cycle and latest events."
      right={<div className="flex items-center gap-2 text-[13px] text-muted-foreground"><span className="size-2 rounded-full bg-success" />System Online</div>}
    >
      <section className="grid gap-4 md:grid-cols-[1.35fr_.65fr]">
        <Card>
          <div className="mb-1.5 text-xs text-muted-foreground">Current Assembly</div>
          <div className="flex items-start justify-between gap-3">
            <div className="text-[22px] font-bold">IN PROGRESS</div>
            <Badge tone="success">ACTIVE</Badge>
          </div>
          <div className="mt-5 flex justify-between text-[13px]"><b>State 5 — Blue Block</b><span>5 / 10</span></div>
          <Progress value={50} className="mt-2" />
          <div className="mt-[22px] grid grid-cols-1 gap-3.5 sm:grid-cols-2">
            <Meta label="Cycle Number" value="ASM-00124" />
            <Meta label="Elapsed Time" value="02:34" />
          </div>
        </Card>
        <Card className="flex min-h-[300px] flex-col border-ink bg-ink text-card">
          <div className="flex justify-between text-[13px] text-ink-foreground"><span>Camera Feed</span><span className="text-destructive-soft">● LIVE</span></div>
          <div className="mt-3.5 flex min-h-[235px] flex-1 items-center justify-center rounded-lg border border-dashed border-muted-foreground text-center text-sm text-ink-foreground">
            Camera image placeholder<br />Detection boxes will appear here
          </div>
          <div className="mt-2.5 text-xs text-ink-foreground">Latest frame confidence: 94%</div>
        </Card>
      </section>

      <Card className="mt-4">
        <div className="mb-3.5 font-bold">Signed-in Manager<div className="mt-1 text-xs font-normal text-muted-foreground">Current authenticated account</div></div>
        <div className="flex flex-wrap justify-between gap-4">
          <Meta label="Manager" value="Demo account" />
          <Meta label="Login time" value="—" />
          <Meta label="Logout time" value="—" />
        </div>
      </Card>

      <KpiGrid className="mt-4" items={[
        { label: "Assemblies Today", value: 42, note: "Completed cycles" },
        { label: "Correct", value: 39, note: "Passed assemblies" },
        { label: "Defective", value: 3, note: "Failed cycles" },
        { label: "Pass Rate", value: "93%", note: "Completed cycles" },
      ]} />

      <Card className="mt-4">
        <div className="mb-3.5 flex items-center justify-between">
          <div><div className="font-bold">Live Event Log</div><div className="mt-1 text-xs text-muted-foreground">Latest assembly events</div></div>
          <Btn>Refresh</Btn>
        </div>
        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead><tr>{["Time", "State", "Object", "Confidence", "Event", "Status"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
            <tbody>
              {liveEvents.map((e, i) => {
                const fail = e.status === "FAILED";
                return (
                  <tr key={i} className={cn(fail && "bg-destructive-soft")}>
                    <td className={td}>{e.time}</td><td className={td}>{e.state}</td><td className={td}>{e.object}</td>
                    <td className={td}>{Math.round(e.confidence * 100)}%</td>
                    <td className={cn(td, fail && "text-destructive")}>{e.event}</td>
                    <td className={td}><Badge tone={fail ? "danger" : "success"}>{e.status}</Badge></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </Page>
  );
}
