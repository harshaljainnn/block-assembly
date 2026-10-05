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
import { cn } from "@/lib/utils";
import { LiveFeed } from "@/components/assembly/live-feed";
import { supabase } from "@/integrations/supabase/client";
import type { Database } from "@/integrations/supabase/types";

export const Route = createFileRoute("/_authenticated/")({
  head: () => ({
    meta: [
      { title: "Live Monitor — Assembly Monitor" },
      {
        name: "description",
        content:
          "Live view of the current assembly cycle and latest events.",
      },
      { property: "og:title", content: "Live Monitor — Assembly Monitor" },
      {
        property: "og:description",
        content:
          "Live view of the current assembly cycle and latest events.",
      },
    ],
  }),
  component: LiveMonitor,
});

type AssemblySession =
  Database["public"]["Tables"]["assembly_sessions"]["Row"];

type AssemblyEvent =
  Database["public"]["Tables"]["assembly_events"]["Row"];

type Operator = Database["public"]["Tables"]["operators"]["Row"];

type OperatorSession =
  Database["public"]["Tables"]["operator_sessions"]["Row"];

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
    </div>
  );
}

function formatElapsed(seconds: number) {
  const safeSeconds = Math.max(0, Math.floor(seconds));

  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const secs = safeSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0",
    )}:${String(secs).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0",
  )}`;
}

function formatTime(timestamp: string | null) {
  if (!timestamp) return "—";

  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatIncomingObject(value: string | null) {
  if (!value) return "—";

  try {
    const parsed: unknown = JSON.parse(value);

    if (typeof parsed === "string") {
      return parsed;
    }

    if (typeof parsed === "object" && parsed !== null) {
      const object = parsed as Record<string, unknown>;

      if (typeof object["name"] === "string") return object["name"];
      if (typeof object["object"] === "string") return object["object"];
      if (typeof object["label"] === "string") return object["label"];
    }

    return JSON.stringify(parsed);
  } catch {
    return value;
  }
}

function formatStateName(name: string | null) {
  if (!name) return "—";
  return name
    .replace(/^state_\d+_?/, (match) => `State ${match.replace(/\D/g, "")}: `)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function eventStatusLabel(status: string) {
  switch (status) {
    case "advanced":
      return "ADVANCED";
    case "completed":
      return "COMPLETED";
    case "error":
      return "ERROR";
    case "holding":
      return "HOLDING";
    default:
      return status.toUpperCase();
  }
}

function LiveMonitor() {
  const { user, isManager, profile } = Route.useRouteContext();

  const [assembly, setAssembly] = useState<AssemblySession | null>(null);
  const [events, setEvents] = useState<AssemblyEvent[]>([]);
  const [operator, setOperator] = useState<Operator | null>(null);
  const [operatorsList, setOperatorsList] = useState<Operator[]>([]);
  const [selectedOperatorId, setSelectedOperatorId] = useState<string>("all");
  const [operatorSession, setOperatorSession] =
    useState<OperatorSession | null>(null);

  const [todaySessions, setTodaySessions] = useState<AssemblySession[]>([]);
  const [mySessions, setMySessions] = useState<AssemblySession[]>([]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [now, setNow] = useState(Date.now());

  const loadLiveData = useCallback(async () => {
    try {
      setRefreshing(true);

      // Fetch list of all registered operators
      const { data: allOperators } = await supabase
        .from("operators")
        .select("*")
        .order("name", { ascending: true });
      setOperatorsList(allOperators ?? []);

      /*
       * 1. Active or latest assembly
       * First check if there is an in-progress session. If not, pick the latest.
       */
      let inProgressQuery = supabase
        .from("assembly_sessions")
        .select("*")
        .eq("status", "in_progress")
        .order("start_time", { ascending: false });

      if (selectedOperatorId !== "all") {
        inProgressQuery = inProgressQuery.eq("operator_id", selectedOperatorId);
      }

      const { data: inProgressSessions } = await inProgressQuery.limit(1);

      let targetAssembly: AssemblySession | null =
        inProgressSessions?.[0] ?? null;

      if (!targetAssembly) {
        let latestQuery = supabase
          .from("assembly_sessions")
          .select("*")
          .order("start_time", { ascending: false });

        if (selectedOperatorId !== "all") {
          latestQuery = latestQuery.eq("operator_id", selectedOperatorId);
        }

        const { data: latestSessions } = await latestQuery.limit(1);
        targetAssembly = latestSessions?.[0] ?? null;
      }

      setAssembly(targetAssembly);

      /*
       * 2. Live Events Feed
       */
      let fetchedEvents: AssemblyEvent[] = [];
      if (targetAssembly) {
        const { data: sessionEvents } = await supabase
          .from("assembly_events")
          .select("*")
          .eq("assembly_id", targetAssembly.id)
          .order("timestamp", { ascending: false })
          .limit(30);

        if (sessionEvents && sessionEvents.length > 0) {
          fetchedEvents = sessionEvents;
        }
      }

      if (fetchedEvents.length === 0) {
        const { data: globalEvents } = await supabase
          .from("assembly_events")
          .select("*")
          .order("timestamp", { ascending: false })
          .limit(30);

        fetchedEvents = globalEvents ?? [];
      }

      setEvents(fetchedEvents);

      /*
       * 3. Today's assemblies
       */
      const startOfToday = new Date();
      startOfToday.setHours(0, 0, 0, 0);

      let todayQuery = supabase
        .from("assembly_sessions")
        .select("*")
        .gte("start_time", startOfToday.toISOString())
        .order("start_time", { ascending: false });

      if (selectedOperatorId !== "all") {
        todayQuery = todayQuery.eq("operator_id", selectedOperatorId);
      }

      const { data: sessionsToday, error: todayError } = await todayQuery;

      if (todayError) {
        console.error("Failed to load today's assemblies:", todayError);
        setTodaySessions([]);
      } else {
        setTodaySessions(sessionsToday ?? []);
      }

      /*
       * 4. Current operator for the active assembly
       */
      let currentOperator: Operator | null = null;
      if (targetAssembly?.operator_id) {
        currentOperator =
          allOperators?.find((op) => op.id === targetAssembly.operator_id) ??
          null;
      }

      if (!currentOperator && selectedOperatorId !== "all") {
        currentOperator =
          allOperators?.find((op) => op.id === selectedOperatorId) ?? null;
      }

      if (!currentOperator) {
        currentOperator = allOperators?.[0] ?? null;
      }

      setOperator(currentOperator ?? null);

      /*
       * 5. Current operator's today's assemblies
       */
      if (currentOperator) {
        const { data: ownSessions } = await supabase
          .from("assembly_sessions")
          .select("*")
          .eq("operator_id", currentOperator.id)
          .gte("start_time", startOfToday.toISOString())
          .order("start_time", { ascending: false });

        setMySessions(ownSessions ?? []);

        /*
         * 6. Latest login session
         */
        const { data: latestOperatorSession } = await supabase
          .from("operator_sessions")
          .select("*")
          .eq("operator_id", currentOperator.id)
          .order("login_time", { ascending: false })
          .limit(1)
          .maybeSingle();

        setOperatorSession(latestOperatorSession ?? null);
      } else {
        setMySessions([]);
        setOperatorSession(null);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [user?.id, selectedOperatorId]);

  /*
   * Initial load + realtime subscriptions + heartbeat polling
   */
  useEffect(() => {
    void loadLiveData();

    const channel = supabase
      .channel("dashboard-live-channel")
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "assembly_sessions",
        },
        () => {
          void loadLiveData();
        },
      )
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "assembly_events",
        },
        () => {
          void loadLiveData();
        },
      )
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "operators",
        },
        () => {
          void loadLiveData();
        },
      )
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "operator_sessions",
        },
        () => {
          void loadLiveData();
        },
      )
      .subscribe();

    // Heartbeat poll every 1 second to guarantee instant live updates
    const interval = window.setInterval(() => {
      void loadLiveData();
    }, 1000);

    return () => {
      void supabase.removeChannel(channel);
      window.clearInterval(interval);
    };
  }, [loadLiveData]);

  /*
   * Update elapsed time every second while assembly is running.
   */
  useEffect(() => {
    if (assembly?.status !== "in_progress") return;

    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => window.clearInterval(interval);
  }, [assembly?.status]);

  /*
   * Latest event is first because events are ordered descending.
   */
  const latestEvent = events[0] ?? null;

  /*
   * Current assembly information
   */
  const isInProgress = assembly?.status === "in_progress";

  const elapsedSeconds = useMemo(() => {
    if (!assembly) return 0;

    if (assembly.duration_seconds !== null) {
      return assembly.duration_seconds;
    }

    const start = new Date(assembly.start_time).getTime();

    if (!Number.isFinite(start)) return 0;

    return Math.max(0, (now - start) / 1000);
  }, [assembly, now]);

  const currentStepNumber = Math.max(
    assembly?.states_completed || 0,
    latestEvent?.state_index || 0,
  );

  const totalStatesNumber = Math.max(1, assembly?.total_states || 9);

  const progress = assembly
    ? Math.min(
      100,
      Math.round((currentStepNumber / totalStatesNumber) * 100),
    )
    : 0;

  const currentState = latestEvent
    ? formatStateName(
      latestEvent.state_title ||
      latestEvent.state_name ||
      `State ${latestEvent.state_index ?? "—"}`,
    )
    : assembly
      ? `State ${assembly.states_completed || 0}`
      : "No active assembly";

  /*
   * Today's KPI values
   */
  const completedToday = todaySessions.filter(
    (session) => session.status === "pass" || session.status === "fail",
  );

  const passedToday = todaySessions.filter(
    (session) => session.status === "pass",
  ).length;

  const failedToday = todaySessions.filter(
    (session) => session.status === "fail",
  ).length;

  const passRate =
    completedToday.length > 0
      ? Math.round((passedToday / completedToday.length) * 100)
      : 0;

  /*
   * Current operator's KPI values
   */
  const myCompletedSessions = mySessions.filter(
    (session) => session.status === "pass" || session.status === "fail",
  );

  const myPassedSessions = mySessions.filter(
    (session) => session.status === "pass",
  ).length;

  const myPassRate =
    myCompletedSessions.length > 0
      ? Math.round((myPassedSessions / myCompletedSessions.length) * 100)
      : 0;

  const dailyGoal = operator?.goal_per_day ?? profile?.goal_per_day ?? 10;

  const goalProgress =
    dailyGoal > 0
      ? Math.min(100, Math.round((myCompletedSessions.length / dailyGoal) * 100))
      : 0;

  const handleStopCycle = async () => {
    if (!assembly || assembly.status !== "in_progress") return;

    try {
      const nowIso = new Date().toISOString();
      const calculatedDuration = Math.max(
        1,
        Math.floor((Date.now() - new Date(assembly.start_time).getTime()) / 1000),
      );

      await supabase
        .from("assembly_sessions")
        .update({
          status: "fail",
          failure_reason: "Inspection stopped by user",
          end_time: nowIso,
          duration_seconds: calculatedDuration,
        })
        .eq("id", assembly.id);

      await loadLiveData();
    } catch (err) {
      console.error("Failed to stop cycle:", err);
    }
  };

  return (
    <Page
      title="Live Monitor"
      subtitle="Current assembly cycle and latest events."
      right={
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground font-medium">Filter Operator:</span>
            <select
              value={selectedOperatorId}
              onChange={(e) => setSelectedOperatorId(e.target.value)}
              className="h-8 rounded-md border border-border bg-background px-2 text-xs outline-none focus:border-ink cursor-pointer"
            >
              <option value="all">All Operators (Plant-wide)</option>
              {operatorsList.map((op) => (
                <option key={op.id} value={op.id}>
                  {op.name} {op.operator_code ? `(${op.operator_code})` : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="size-2 rounded-full bg-success animate-pulse" />
            Live Sync
          </div>
        </div>
      }
    >
      <LiveFeed className="mb-4" />

      <section>
        <Card>
          <div className="mb-1.5 text-xs text-muted-foreground">
            Current Assembly
          </div>

          <div className="flex items-start justify-between gap-3">
            <div className="text-[22px] font-bold">
              {loading
                ? "LOADING..."
                : isInProgress
                  ? "IN PROGRESS"
                  : assembly?.status === "pass"
                    ? "COMPLETED"
                    : assembly?.status === "fail"
                      ? "FAILED"
                      : "NO ACTIVE ASSEMBLY"}
            </div>

            <div className="flex items-center gap-2">
              {isInProgress && (
                <button
                  type="button"
                  onClick={() => void handleStopCycle()}
                  className="rounded border border-destructive/40 bg-destructive/10 px-2.5 py-1 text-xs font-semibold text-destructive hover:bg-destructive hover:text-white transition-colors"
                >
                  Stop Cycle
                </button>
              )}

              <Badge
                tone={
                  assembly?.status === "fail"
                    ? "danger"
                    : isInProgress
                      ? "success"
                      : "success"
                }
              >
                {isInProgress
                  ? "ACTIVE"
                  : assembly?.status?.toUpperCase() ?? "IDLE"}
              </Badge>
            </div>
          </div>

          <div className="mt-5 flex justify-between text-[13px]">
            <b>{currentState}</b>

            <span>
              {assembly
                ? `${assembly.states_completed} / ${assembly.total_states}`
                : "—"}
            </span>
          </div>

          <Progress value={progress} className="mt-2" />

          <div className="mt-[22px] grid grid-cols-1 gap-3.5 sm:grid-cols-2 md:grid-cols-4">
            <Meta
              label="Cycle Number"
              value={assembly?.cycle_id ?? "—"}
            />

            <Meta
              label="Assigned Operator"
              value={
                operator
                  ? `${operator.name}${operator.operator_code ? ` (${operator.operator_code})` : ""}`
                  : "Unassigned"
              }
            />

            <Meta
              label="Elapsed Time"
              value={formatElapsed(elapsedSeconds)}
            />

            <Meta
              label="Latest State"
              value={formatStateName(
                latestEvent?.state_title ||
                  latestEvent?.state_name ||
                  (latestEvent?.state_index
                    ? `State ${latestEvent.state_index}`
                    : "—"),
              )}
            />
          </div>
        </Card>
      </section>


      <KpiGrid
        className="mt-4"
        items={[
          {
            label: "Assemblies Today",
            value: completedToday.length,
            note: "Completed cycles",
          },
          {
            label: "Correct",
            value: passedToday,
            note: "Passed assemblies",
          },
          {
            label: "Defective",
            value: failedToday,
            note: "Failed cycles",
          },
          {
            label: "Pass Rate",
            value: `${passRate}%`,
            note: "Completed cycles",
          },
        ]}
      />


      <Card className="mt-4">
        <div className="mb-3.5 flex items-center justify-between">
          <div>
            <div className="font-bold">Live Event Log</div>

            <div className="mt-1 text-xs text-muted-foreground">
              Latest assembly events
            </div>
          </div>

          <Btn onClick={() => void loadLiveData()}>
            {refreshing ? "Refreshing..." : "Refresh"}
          </Btn>
        </div>

        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                {[
                  "Time",
                  "State",
                  "Object",
                  "Confidence",
                  "Event",
                  "Status",
                ].map((header) => (
                  <th key={header} className={th}>
                    {header}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {events.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
                    className={`${td} py-8 text-center text-muted-foreground`}
                  >
                    {loading
                      ? "Loading events..."
                      : "No assembly events yet."}
                  </td>
                </tr>
              ) : (
                events.map((event) => {
                  const failed = event.status === "error";

                  return (
                    <tr
                      key={event.id}
                      className={cn(
                        failed && "bg-destructive-soft",
                      )}
                    >
                      <td className={td}>
                        {formatTime(event.timestamp)}
                      </td>

                      <td className={td}>
                        {formatStateName(
                          event.state_title ||
                            event.state_name ||
                            (event.state_index != null
                              ? `State ${event.state_index}`
                              : "—"),
                        )}
                      </td>

                      <td className={td}>
                        {formatIncomingObject(event.incoming_object)}
                      </td>

                      <td className={td}>
                        {event.confidence != null
                          ? `${Math.round(event.confidence * 100)}%`
                          : "—"}
                      </td>

                      <td
                        className={cn(
                          td,
                          failed && "text-destructive",
                        )}
                      >
                        {event.state_title ||
                          event.diagnostic ||
                          event.consensus ||
                          event.state_name ||
                          eventStatusLabel(event.status)}
                      </td>

                      <td className={td}>
                        <Badge
                          tone={failed ? "danger" : "success"}
                        >
                          {eventStatusLabel(event.status)}
                        </Badge>
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