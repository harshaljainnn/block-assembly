import { createFileRoute } from "@tanstack/react-router";
import { Card, KpiGrid, Page, Progress, inputCls } from "@/components/assembly/ui";
import { analyticsRows, fmtDuration } from "@/lib/assembly-data";

export const Route = createFileRoute("/_authenticated/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — Assembly Monitor" },
      { name: "description", content: "Combined assembly quality, throughput and process bottleneck analytics." },
      { property: "og:title", content: "Analytics — Assembly Monitor" },
      { property: "og:description", content: "Combined assembly quality, throughput and process bottleneck analytics." },
    ],
  }),
  component: Analytics,
});

const hours = ["08 AM", "10 AM", "12 PM", "02 PM", "04 PM", "06 PM"];
const counts = [5, 8, 7, 10, 7, 5];
const stateVals: Record<string, number> = { "State 1": 18, "State 2": 21, "State 3": 19, "State 4": 24, "State 5": 31, "State 6": 28 };

function ChartCard({ title, sub, children }: { title: string; sub: string; children: React.ReactNode }) {
  return <Card><div className="font-bold">{title}</div><div className="mt-1 text-xs text-muted-foreground">{sub}</div>{children}</Card>;
}

function Analytics() {
  const rows = analyticsRows;
  const passed = rows.filter((r) => r.status === "Pass");
  const failed = rows.filter((r) => r.status === "Failed");
  const times = rows.map((r) => r.duration);
  const avg = Math.round(times.reduce((a, b) => a + b, 0) / times.length);
  const rate = Math.round((passed.length / rows.length) * 100);
  const maxCount = Math.max(...counts);
  const vals = rows.slice(-10).map((r) => (r.status === "Failed" ? 0 : 100));
  const points = vals.map((v, i) => `${30 + i * (440 / Math.max(1, vals.length - 1))},${200 - v * 0.9}`).join(" ");
  const reasons: Record<string, number> = {};
  failed.forEach((r) => { const k = r.failure_reason || "Other"; reasons[k] = (reasons[k] || 0) + 1; });
  const maxState = Math.max(...Object.values(stateVals));
  const bottleneck = Object.entries(stateVals).sort((a, b) => b[1] - a[1])[0]!;

  return (
    <Page
      title="Analytics"
      subtitle="All Operators — combined assembly quality, productivity and process performance."
      right={<select className={`${inputCls} !min-w-[120px]`}><option>Today</option><option>7 Days</option><option>30 Days</option></select>}
    >
      <KpiGrid items={[
        { label: "Total Assemblies", value: rows.length, note: "Completed cycles" },
        { label: "First-Pass Yield", value: `${rate}%`, note: "Passed without failure" },
        { label: "Average Cycle Time", value: fmtDuration(avg), note: "Per completed assembly" },
        { label: "Throughput", value: `${(3600 / avg).toFixed(1)}/hr`, note: "Estimated completed cycles/hour" },
      ]} />

      <section className="mt-4 grid gap-4 md:grid-cols-2">
        <ChartCard title="Daily Throughput" sub="Completed assemblies by period">
          <div className="flex h-[210px] items-end gap-3 px-2 pb-2 pt-5">
            {counts.map((v, i) => (
              <div key={i} className="flex h-full flex-1 flex-col items-center justify-end">
                <div className="w-full max-w-[42px] rounded-t bg-ink" style={{ height: Math.max(8, (v / maxCount) * 160) }} />
                <div className="mt-2 text-center text-[11px] text-muted-foreground">{hours[i]}<br />{v}</div>
              </div>
            ))}
          </div>
        </ChartCard>
        <ChartCard title="Quality Trend" sub="Pass rate across recent cycles">
          <svg className="h-[220px] w-full" viewBox="0 0 500 220" preserveAspectRatio="none">
            <line x1="20" y1="200" x2="480" y2="200" className="stroke-border" />
            <line x1="20" y1="95" x2="480" y2="95" className="stroke-muted" />
            <polyline fill="none" className="stroke-ink" strokeWidth="3" points={points} />
          </svg>
          <div className="mt-2 text-[11px] text-muted-foreground">Pass = 100%, failed cycle = 0%</div>
        </ChartCard>
        <ChartCard title="Defect Breakdown" sub="Where failed assemblies are occurring">
          <div className="mt-[18px]">
            {Object.entries(reasons).map(([k, v]) => (
              <div key={k} className="mb-4">
                <div className="flex justify-between text-[13px]"><span>{k}</span><b>{v}</b></div>
                <Progress className="mt-1.5" barClass="bg-destructive" value={(v / Math.max(1, failed.length)) * 100} />
              </div>
            ))}
          </div>
        </ChartCard>
        <ChartCard title="Process Bottlenecks" sub="Average time spent in each assembly state">
          <div className="mt-[18px]">
            {Object.entries(stateVals).map(([k, v]) => (
              <div key={k} className="mb-3">
                <div className="flex justify-between text-[13px]"><span>{k}</span><span>{v} sec</span></div>
                <Progress className="mt-1" value={(v / maxState) * 100} />
              </div>
            ))}
          </div>
        </ChartCard>
      </section>

      <KpiGrid className="mt-4" items={[
        { label: "Defect Rate", value: `${100 - rate}%`, note: "Failed cycles / total" },
        { label: "Best Cycle", value: fmtDuration(Math.min(...times)), note: "Fastest completed assembly" },
        { label: "Slowest Cycle", value: fmtDuration(Math.max(...times)), note: "Longest completed assembly" },
        { label: "Bottleneck State", value: bottleneck[0], note: `${bottleneck[1]} sec average` },
      ]} />
    </Page>
  );
}
