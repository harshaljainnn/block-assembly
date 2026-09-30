import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { Btn, Card, KpiGrid, Page, inputCls, td, th } from "@/components/assembly/ui";
import { demoOperators, type Operator } from "@/lib/assembly-data";

export const Route = createFileRoute("/_authenticated/operators")({
  head: () => ({
    meta: [
      { title: "Operators — Assembly Monitor" },
      { name: "description", content: "Manager overview of operator activity, daily goals and performance." },
      { property: "og:title", content: "Operators — Assembly Monitor" },
      { property: "og:description", content: "Manager overview of operator activity, daily goals and performance." },
    ],
  }),
  component: Operators,
});

function Operators() {
  const [ops, setOps] = useState<Operator[]>(demoOperators);
  const [goals, setGoals] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<Operator | null>(null);

  const save = (o: Operator) => {
    const g = Number(goals[o.user_id] ?? o.goal_per_day);
    if (!Number.isFinite(g) || g < 1) { toast.error("Enter a valid goal."); return; }
    setOps((p) => p.map((x) => (x.user_id === o.user_id ? { ...x, goal_per_day: Math.round(g) } : x)));
    toast.success(`Goal saved for ${o.name}.`);
  };

  return (
    <Page title="Operators" subtitle="Manager overview of operator activity, goals and performance.">
      <KpiGrid items={[
        { label: "Operators", value: ops.length },
        { label: "Currently Active", value: "—" },
        { label: "Assemblies Today", value: "—" },
        { label: "Overall Pass Rate", value: "—" },
      ]} />
      <Card className="mt-4">
        <div className="font-bold">Operator Activity</div>
        <div className="mb-4 mt-1.5 text-xs text-muted-foreground">Click an operator to view their individual analytics.</div>
        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead><tr>{["Operator", "Daily Goal", "Completed", "Goal Progress", "Actions"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
            <tbody>
              {ops.map((o) => (
                <tr key={o.user_id}>
                  <td className={td}><b>{o.name}</b><div className="mt-1 text-xs text-muted-foreground">{o.email}</div></td>
                  <td className={td}>
                    <input type="number" min={1} className={`${inputCls} !min-w-0 w-[90px]`} value={goals[o.user_id] ?? o.goal_per_day}
                      onChange={(e) => setGoals((g) => ({ ...g, [o.user_id]: e.target.value }))} />
                  </td>
                  <td className={td}>{o.completed}</td>
                  <td className={td}>{Math.round((o.completed / Math.max(1, o.goal_per_day)) * 100)}%</td>
                  <td className={`${td} space-x-1 whitespace-nowrap`}>
                    <Btn onClick={() => save(o)}>Save Goal</Btn>
                    <Btn onClick={() => setSelected(o)}>View Analytics</Btn>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {selected && (
        <Card className="mt-4">
          <div className="mb-3.5 flex items-center justify-between">
            <div><div className="font-bold">{selected.name} — Performance</div><div className="mt-1 text-xs text-muted-foreground">Individual performance analytics</div></div>
            <Btn onClick={() => setSelected(null)}>Close</Btn>
          </div>
          <KpiGrid className="mt-4" items={[
            { label: "Goal", value: `${selected.goal_per_day} assemblies` },
            { label: "Completed", value: selected.completed },
            { label: "Pass Rate", value: `${selected.pass_rate}%` },
            { label: "Avg Cycle Time", value: "—" },
          ]} />
          <div className="mt-4 text-xs text-muted-foreground">Demo operator data. Cycle-time and event-level analytics will populate when this operator is connected to live assembly records.</div>
        </Card>
      )}
    </Page>
  );
}
