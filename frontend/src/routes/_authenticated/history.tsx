import { useCallback, useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Badge,
  Btn,
  Card,
  Page,
  td,
  th,
} from "@/components/assembly/ui";
import { supabase } from "@/integrations/supabase/client";
import type { Database } from "@/integrations/supabase/types";

type AssemblySession =
  Database["public"]["Tables"]["assembly_sessions"]["Row"];

type Operator =
  Database["public"]["Tables"]["operators"]["Row"];

export const Route = createFileRoute("/_authenticated/history")({
  head: () => ({
    meta: [
      { title: "Assembly History — Assembly Monitor" },
      {
        name: "description",
        content: "Historical assembly cycle records.",
      },
    ],
  }),
  component: History,
});

function formatDate(value: string | null) {
  if (!value) return "—";

  return new Date(value).toLocaleString([], {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(seconds: number | null) {
  if (seconds == null) return "—";

  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);

  if (minutes === 0) {
    return `${secs}s`;
  }

  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

function History() {
  const { isManager } = Route.useRouteContext();

  const [sessions, setSessions] = useState<AssemblySession[]>([]);
  const [operators, setOperators] = useState<Operator[]>([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [loading, setLoading] = useState(true);

  const loadHistory = useCallback(async (silent = false) => {
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
      console.error("Failed to load history:", sessionsResult.error);
    } else {
      setSessions(sessionsResult.data ?? []);
    }

    if (operatorsResult.error) {
      console.error("Failed to load operators:", operatorsResult.error);
    } else {
      setOperators(operatorsResult.data ?? []);
    }

    setLoading(false);
  }, []);

  useEffect(() => {
    void loadHistory();

    const channel = supabase
      .channel("history-live-channel")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "assembly_sessions" },
        () => void loadHistory(true),
      )
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "operators" },
        () => void loadHistory(true),
      )
      .subscribe();

    const interval = window.setInterval(() => {
      void loadHistory(true);
    }, 4000);

    return () => {
      void supabase.removeChannel(channel);
      window.clearInterval(interval);
    };
  }, [loadHistory]);

  const operatorMap = useMemo(() => {
    return new Map(operators.map((operator) => [operator.id, operator]));
  }, [operators]);

  const filteredSessions = useMemo(() => {
    const query = search.trim().toLowerCase();

    return sessions.filter((session) => {
      const operator = session.operator_id
        ? operatorMap.get(session.operator_id)
        : null;

      const matchesSearch =
        !query ||
        session.cycle_id.toLowerCase().includes(query) ||
        operator?.name.toLowerCase().includes(query) ||
        operator?.operator_code?.toLowerCase().includes(query);

      const matchesStatus =
        statusFilter === "all" ||
        session.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [sessions, search, statusFilter, operatorMap]);

  return (
    <Page
      title="Assembly History"
      subtitle={
        isManager
          ? "Historical records for all assembly cycles."
          : "Your historical assembly records."
      }
      right={
        <Btn onClick={() => void loadHistory()}>
          Refresh
        </Btn>
      }
    >
      <Card>
        <div className="flex flex-col gap-3 md:flex-row">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search cycle ID or operator..."
            className="h-10 flex-1 rounded-lg border border-border bg-background px-3 text-sm outline-none focus:border-ink"
          />

          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
            className="h-10 rounded-lg border border-border bg-background px-3 text-sm outline-none"
          >
            <option value="all">All statuses</option>
            <option value="in_progress">In progress</option>
            <option value="pass">Passed</option>
            <option value="fail">Failed</option>
          </select>
        </div>
      </Card>

      <Card className="mt-4">
        <div className="mb-4">
          <div className="font-bold">Assembly Records</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {filteredSessions.length} record
            {filteredSessions.length === 1 ? "" : "s"}
          </div>
        </div>

        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                {[
                  "Cycle",
                  "Operator",
                  "Start",
                  "End",
                  "Duration",
                  "Progress",
                  "Status",
                  "Failure Reason",
                ].map((header) => (
                  <th key={header} className={th}>
                    {header}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {loading ? (
                <tr>
                  <td
                    colSpan={8}
                    className={`${td} py-10 text-center text-muted-foreground`}
                  >
                    Loading history...
                  </td>
                </tr>
              ) : filteredSessions.length === 0 ? (
                <tr>
                  <td
                    colSpan={8}
                    className={`${td} py-10 text-center text-muted-foreground`}
                  >
                    No assembly records found.
                  </td>
                </tr>
              ) : (
                filteredSessions.map((session) => {
                  const operator = session.operator_id
                    ? operatorMap.get(session.operator_id)
                    : null;

                  const statusTone =
                    session.status === "fail"
                      ? "danger"
                      : session.status === "pass"
                        ? "success"
                        : "success";

                  return (
                    <tr key={session.id}>
                      <td className={td}>
                        <span className="font-semibold">
                          {session.cycle_id}
                        </span>
                      </td>

                      <td className={td}>
                        {operator?.name ??
                          operator?.operator_code ??
                          "Unassigned"}
                      </td>

                      <td className={td}>
                        {formatDate(session.start_time)}
                      </td>

                      <td className={td}>
                        {formatDate(session.end_time)}
                      </td>

                      <td className={td}>
                        {formatDuration(session.duration_seconds)}
                      </td>

                      <td className={td}>
                        {session.states_completed} /{" "}
                        {session.total_states}
                      </td>

                      <td className={td}>
                        <Badge tone={statusTone}>
                          {session.status.toUpperCase()}
                        </Badge>
                      </td>

                      <td
                        className={`${td} max-w-[220px] truncate`}
                        title={session.failure_reason ?? ""}
                      >
                        {session.failure_reason ?? "—"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </Page>
  );
}