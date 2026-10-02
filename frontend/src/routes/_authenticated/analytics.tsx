import { useCallback, useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Badge,
  Btn,
  Card,
  KpiGrid,
  Page,
  Progress,
  td,
  th,
} from "@/components/assembly/ui";
import { supabase } from "@/integrations/supabase/client";
import type { Database } from "@/integrations/supabase/types";

type AssemblySession =
  Database["public"]["Tables"]["assembly_sessions"]["Row"];

type Operator =
  Database["public"]["Tables"]["operators"]["Row"];

export const Route = createFileRoute("/_authenticated/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — Assembly Monitor" },
      {
        name: "description",
        content: "Assembly performance analytics.",
      },
    ],
  }),
  component: Analytics,
});

function Analytics() {
  const { isManager, operator, profile } = Route.useRouteContext();

  const [sessions, setSessions] = useState<AssemblySession[]>([]);
  const [operators, setOperators] = useState<Operator[]>([]);
  const [loading, setLoading] = useState(true);

  const loadAnalytics = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);

    const [sessionsResult, operatorsResult] = await Promise.all([
      supabase
        .from("assembly_sessions")
        .select("*")
        .order("start_time", { ascending: false }),

      supabase
        .from("operators")
        .select("*")
        .order("name", { ascending: true }),
    ]);

    if (sessionsResult.error) {
      console.error(
        "Failed to load analytics:",
        sessionsResult.error,
      );
    } else {
      setSessions(sessionsResult.data ?? []);
    }

    if (operatorsResult.error) {
      console.error(
        "Failed to load operators:",
        operatorsResult.error,
      );
    } else {
      setOperators(operatorsResult.data ?? []);
    }

    setLoading(false);
  }, []);

  useEffect(() => {
    void loadAnalytics();

    const channel = supabase
      .channel("analytics-live-channel")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "assembly_sessions" },
        () => void loadAnalytics(true),
      )
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "operators" },
        () => void loadAnalytics(true),
      )
      .subscribe();

    const interval = window.setInterval(() => {
      void loadAnalytics(true);
    }, 4000);

    return () => {
      void supabase.removeChannel(channel);
      window.clearInterval(interval);
    };
  }, [loadAnalytics]);

  const completed = useMemo(
    () =>
      sessions.filter(
        (session) =>
          session.status === "pass" ||
          session.status === "fail",
      ),
    [sessions],
  );

  const passed = completed.filter(
    (session) => session.status === "pass",
  ).length;

  const failed = completed.filter(
    (session) => session.status === "fail",
  ).length;

  const passRate =
    completed.length > 0
      ? Math.round((passed / completed.length) * 100)
      : 0;

  const averageDuration =
    completed.length > 0
      ? Math.round(
        completed.reduce(
          (sum, session) =>
            sum + (session.duration_seconds ?? 0),
          0,
        ) / completed.length,
      )
      : 0;

  const todayCount = useMemo(() => {
    const start = new Date();
    start.setHours(0, 0, 0, 0);

    return sessions.filter(
      (session) =>
        new Date(session.start_time) >= start &&
        (session.status === "pass" ||
          session.status === "fail"),
    ).length;
  }, [sessions]);

  const operatorMap = useMemo(
    () =>
      new Map(
        operators.map((operator) => [operator.id, operator]),
      ),
    [operators],
  );

  const operatorStats = useMemo(() => {
    return operators.map((operator) => {
      const own = completed.filter(
        (session) => session.operator_id === operator.id,
      );

      const ownPassed = own.filter(
        (session) => session.status === "pass",
      ).length;

      const ownFailed = own.filter(
        (session) => session.status === "fail",
      ).length;

      const ownPassRate =
        own.length > 0
          ? Math.round((ownPassed / own.length) * 100)
          : 0;

      const avgTime =
        own.length > 0
          ? Math.round(
            own.reduce(
              (sum, session) =>
                sum + (session.duration_seconds ?? 0),
              0,
            ) / own.length,
          )
          : 0;

      return {
        operator,
        total: own.length,
        passed: ownPassed,
        failed: ownFailed,
        passRate: ownPassRate,
        averageDuration: avgTime,
      };
    });
  }, [operators, completed]);

  const maxOperatorCount = Math.max(
    1,
    ...operatorStats.map((item) => item.total),
  );

  function formatDuration(seconds: number) {
    if (seconds <= 0) return "—";

    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;

    return minutes > 0
      ? `${minutes}m ${String(secs).padStart(2, "0")}s`
      : `${secs}s`;
  }

  return (
    <Page
      title="Analytics"
      subtitle="Assembly performance and operator metrics."
      right={
        <Btn onClick={() => void loadAnalytics()}>
          Refresh
        </Btn>
      }
    >
      {loading ? (
        <Card>
          <div className="py-8 text-center text-sm text-muted-foreground">
            Loading analytics...
          </div>
        </Card>
      ) : (
        <>
          <KpiGrid
            items={[
              {
                label: "Total Assemblies",
                value: completed.length,
                note: "Completed cycles",
              },
              {
                label: "Passed",
                value: passed,
                note: "Successful cycles",
              },
              {
                label: "Failed",
                value: failed,
                note: "Defective cycles",
              },
              {
                label: "Pass Rate",
                value: `${passRate}%`,
                note: "Overall",
              },
            ]}
          />

          <section className="mt-4 grid gap-4 md:grid-cols-2">
            <Card>
              <div className="font-bold">Production Summary</div>

              <div className="mt-1 text-xs text-muted-foreground">
                Overall assembly performance.
              </div>

              <div className="mt-6 space-y-5">
                <div>
                  <div className="mb-2 flex justify-between text-sm">
                    <span>Pass rate</span>
                    <b>{passRate}%</b>
                  </div>

                  <Progress value={passRate} />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[11px] text-muted-foreground">
                      Average Cycle Time
                    </div>

                    <div className="mt-1 text-xl font-bold">
                      {formatDuration(averageDuration)}
                    </div>
                  </div>

                  <div>
                    <div className="text-[11px] text-muted-foreground">
                      Today's Assemblies
                    </div>

                    <div className="mt-1 text-xl font-bold">
                      {todayCount}
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            <Card>
              <div className="font-bold">Status Breakdown</div>

              <div className="mt-1 text-xs text-muted-foreground">
                Distribution of completed cycles.
              </div>

              <div className="mt-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge tone="success">PASS</Badge>
                    <span className="text-sm">Successful</span>
                  </div>

                  <b>{passed}</b>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge tone="danger">FAIL</Badge>
                    <span className="text-sm">Failed</span>
                  </div>

                  <b>{failed}</b>
                </div>
              </div>
            </Card>
          </section>

          {isManager && (
            <Card className="mt-4">
              <div className="mb-4">
                <div className="font-bold">Operator Performance</div>

                <div className="mt-1 text-xs text-muted-foreground">
                  Performance comparison across operators.
                </div>
              </div>

              <div className="overflow-auto">
                <table className="w-full border-collapse text-[13px]">
                  <thead>
                    <tr>
                      {[
                        "Operator",
                        "Goal",
                        "Completed",
                        "Passed",
                        "Failed",
                        "Pass Rate",
                        "Avg. Time",
                      ].map((header) => (
                        <th key={header} className={th}>
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody>
                    {operatorStats.length === 0 ? (
                      <tr>
                        <td
                          colSpan={7}
                          className={`${td} py-8 text-center text-muted-foreground`}
                        >
                          No operator data available.
                        </td>
                      </tr>
                    ) : (
                      operatorStats.map((item) => (
                        <tr key={item.operator.id}>
                          <td className={td}>
                            <div className="font-semibold">
                              {item.operator.name}
                            </div>

                            <div className="text-xs text-muted-foreground">
                              {item.operator.operator_code ?? "—"}
                            </div>
                          </td>

                          <td className={td}>
                            {item.operator.goal_per_day}
                          </td>

                          <td className={td}>
                            <div className="min-w-[100px]">
                              <div className="flex justify-between">
                                <span>{item.total}</span>
                                <span className="text-xs text-muted-foreground">
                                  {item.operator.goal_per_day > 0
                                    ? Math.min(
                                      100,
                                      Math.round(
                                        (item.total /
                                          item.operator
                                            .goal_per_day) *
                                        100,
                                      ),
                                    )
                                    : 0}
                                  %
                                </span>
                              </div>

                              <Progress
                                value={
                                  item.operator.goal_per_day > 0
                                    ? Math.min(
                                      100,
                                      Math.round(
                                        (item.total /
                                          item.operator
                                            .goal_per_day) *
                                        100,
                                      ),
                                    )
                                    : 0
                                }
                                className="mt-1"
                              />
                            </div>
                          </td>

                          <td className={td}>{item.passed}</td>
                          <td className={td}>{item.failed}</td>
                          <td className={td}>
                            {item.total > 0
                              ? `${item.passRate}%`
                              : "—"}
                          </td>
                          <td className={td}>
                            {formatDuration(
                              item.averageDuration,
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {!isManager && (
            <Card className="mt-4">
              <div className="font-bold">Performance Trend</div>

              <div className="mt-1 text-xs text-muted-foreground">
                Your completed assembly records.
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <div>
                  <div className="text-[11px] text-muted-foreground">
                    Completed
                  </div>
                  <div className="mt-1 text-xl font-bold">
                    {completed.filter(
                      (session) =>
                        session.operator_id ===
                        operator?.id,
                    ).length}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-muted-foreground">
                    Goal
                  </div>
                  <div className="mt-1 text-xl font-bold">
                    {profile?.goal_per_day ??
                      10}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-muted-foreground">
                    Total Operators
                  </div>
                  <div className="mt-1 text-xl font-bold">
                    {operatorMap.size}
                  </div>
                </div>
              </div>
            </Card>
          )}
        </>
      )}
    </Page>
  );
}