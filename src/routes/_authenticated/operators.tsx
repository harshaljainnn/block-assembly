import { useCallback, useEffect, useState } from "react";
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

type Operator =
  Database["public"]["Tables"]["operators"]["Row"];

export const Route = createFileRoute("/_authenticated/operators")({
  head: () => ({
    meta: [
      { title: "Operators — Assembly Monitor" },
      {
        name: "description",
        content: "Operator management and daily assembly goals.",
      },
    ],
  }),
  component: Operators,
});

function Operators() {
  const { isManager } = Route.useRouteContext();

  const [operators, setOperators] = useState<Operator[]>([]);
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [editingGoal, setEditingGoal] = useState<
    Record<string, string>
  >({});

  const [activeOperatorIds, setActiveOperatorIds] = useState<Set<string>>(new Set());
  const [todayOperatorIds, setTodayOperatorIds] = useState<Set<string>>(new Set());

  // Add operator modal state
  const [showAddModal, setShowAddModal] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newCode, setNewCode] = useState("");
  const [newRole, setNewRole] = useState<"operator" | "manager">("operator");
  const [newGoal, setNewGoal] = useState("15");
  const [addingBusy, setAddingBusy] = useState(false);

  const loadOperators = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);

    const startOfToday = new Date();
    startOfToday.setHours(0, 0, 0, 0);

    const [operatorsResult, activeSessionsResult, todaySessionsResult] =
      await Promise.all([
        supabase
          .from("operators")
          .select("*")
          .order("role", { ascending: true })
          .order("name", { ascending: true }),

        supabase
          .from("assembly_sessions")
          .select("operator_id")
          .eq("status", "in_progress"),

        supabase
          .from("assembly_sessions")
          .select("operator_id")
          .gte("start_time", startOfToday.toISOString()),
      ]);

    if (operatorsResult.error) {
      console.error("Failed to load operators:", operatorsResult.error);
    } else {
      setOperators(operatorsResult.data ?? []);

      setEditingGoal((current) => {
        const goals: Record<string, string> = { ...current };

        for (const operator of operatorsResult.data ?? []) {
          if (!(operator.id in goals)) {
            goals[operator.id] = String(operator.goal_per_day);
          }
        }

        return goals;
      });
    }

    if (activeSessionsResult.data) {
      setActiveOperatorIds(
        new Set(
          activeSessionsResult.data
            .map((s) => s.operator_id)
            .filter(Boolean) as string[],
        ),
      );
    }

    if (todaySessionsResult.data) {
      setTodayOperatorIds(
        new Set(
          todaySessionsResult.data
            .map((s) => s.operator_id)
            .filter(Boolean) as string[],
        ),
      );
    }

    setLoading(false);
  }, []);

  useEffect(() => {
    void loadOperators();

    const channel = supabase
      .channel("operators-live-channel")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "operators" },
        () => void loadOperators(true),
      )
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "assembly_sessions" },
        () => void loadOperators(true),
      )
      .subscribe();

    const interval = window.setInterval(() => {
      void loadOperators(true);
    }, 3000);

    return () => {
      void supabase.removeChannel(channel);
      window.clearInterval(interval);
    };
  }, [loadOperators]);

  async function saveGoal(operator: Operator) {
    const value = Number(editingGoal[operator.id]);

    if (!Number.isInteger(value) || value <= 0) {
      window.alert("Goal must be a positive whole number.");
      return;
    }

    setSavingId(operator.id);

    const { error } = await supabase
      .from("operators")
      .update({
        goal_per_day: value,
      })
      .eq("id", operator.id);

    if (error) {
      console.error("Failed to update goal:", error);
      window.alert("Failed to update the operator goal.");
    } else {
      setOperators((current) =>
        current.map((item) =>
          item.id === operator.id
            ? { ...item, goal_per_day: value }
            : item,
        ),
      );
    }

    setSavingId(null);
  }

  async function handleAddOperator(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;

    setAddingBusy(true);
    const goalVal = Number(newGoal) || 15;

    const { error } = await supabase
      .from("operators")
      .insert({
        name: newName.trim(),
        email: newEmail.trim().toLowerCase() || null,
        operator_code: newCode.trim().toUpperCase() || null,
        role: newRole,
        goal_per_day: goalVal,
      });

    if (error) {
      console.error("Failed to add operator:", error);
      window.alert(`Failed to add operator: ${error.message}`);
    } else {
      setNewName("");
      setNewEmail("");
      setNewCode("");
      setNewGoal("15");
      setShowAddModal(false);
      void loadOperators();
    }

    setAddingBusy(false);
  }

  return (
    <Page
      title="Operators"
      subtitle="Operator directory and daily assembly performance goals."
      right={
        <div className="flex items-center gap-2">
          {isManager && (
            <button
              type="button"
              onClick={() => setShowAddModal(true)}
              className="inline-flex items-center justify-center rounded-lg bg-ink px-3 py-1.5 text-xs font-semibold text-card transition-opacity hover:opacity-90 cursor-pointer"
            >
              + Add Operator
            </button>
          )}
          <Btn onClick={() => void loadOperators()}>
            Refresh
          </Btn>
        </div>
      }
    >
      <Card>
        <div className="mb-4">
          <div className="font-bold">Operator Directory</div>

          <div className="mt-1 text-xs text-muted-foreground">
            {operators.length} operator
            {operators.length === 1 ? "" : "s"} registered.
          </div>
        </div>

        <div className="overflow-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                {[
                  "Operator",
                  "Code",
                  "Email",
                  "Role",
                  "Daily Goal",
                  "Status",
                  ...(isManager ? ["Action"] : []),
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
                    colSpan={isManager ? 7 : 6}
                    className={`${td} py-10 text-center text-muted-foreground`}
                  >
                    Loading operators...
                  </td>
                </tr>
              ) : operators.length === 0 ? (
                <tr>
                  <td
                    colSpan={isManager ? 7 : 6}
                    className={`${td} py-10 text-center text-muted-foreground`}
                  >
                    No operators found.
                  </td>
                </tr>
              ) : (
                operators.map((operator) => {
                  return (
                    <tr key={operator.id}>
                      <td className={td}>
                        <div className="font-semibold">
                          {operator.name}
                        </div>
                      </td>

                      <td className={td}>
                        {operator.operator_code ?? "—"}
                      </td>

                      <td className={td}>
                        {operator.email ?? "—"}
                      </td>

                      <td className={td}>
                        <Badge
                          tone={
                            operator.role === "manager"
                              ? "success"
                              : "success"
                          }
                        >
                          {operator.role.toUpperCase()}
                        </Badge>
                      </td>

                      <td className={td}>
                        {isManager ? (
                          <input
                            type="number"
                            min={1}
                            value={
                              editingGoal[operator.id] ??
                              String(operator.goal_per_day)
                            }
                            onChange={(event) =>
                              setEditingGoal((current) => ({
                                ...current,
                                [operator.id]:
                                  event.target.value,
                              }))
                            }
                            className="h-9 w-24 rounded-md border border-border bg-background px-2 text-sm outline-none focus:border-ink"
                          />
                        ) : (
                          operator.goal_per_day
                        )}
                      </td>

                      <td className={td}>
                        {(() => {
                          const isWorking = activeOperatorIds.has(operator.id);
                          const isTodayActive = todayOperatorIds.has(operator.id);

                          if (isWorking) {
                            return (
                              <Badge tone="success">
                                ACTIVE
                              </Badge>
                            );
                          }

                          if (isTodayActive) {
                            return (
                              <Badge tone="warning">
                                IDLE
                              </Badge>
                            );
                          }

                          return (
                            <Badge tone="danger">
                              OFFLINE
                            </Badge>
                          );
                        })()}
                      </td>

                      {isManager && (
                        <td className={td}>
                          <Btn
                            onClick={() =>
                              void saveGoal(operator)
                            }
                          >
                            {savingId === operator.id
                              ? "Saving..."
                              : "Save"}
                          </Btn>
                        </td>
                      )}
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Add Operator Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-xl animate-in fade-in zoom-in-95">
            <h3 className="text-lg font-bold">Add New Operator</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Register an operator to attribute assembly cycles and track daily targets.
            </p>

            <form onSubmit={handleAddOperator} className="mt-5 space-y-3.5">
              <div>
                <label className="block text-xs font-semibold text-muted-foreground mb-1">
                  Full Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. John Doe"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus:border-ink"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-muted-foreground mb-1">
                  Work Email
                </label>
                <input
                  type="email"
                  placeholder="e.g. john.doe@company.com"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus:border-ink"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-muted-foreground mb-1">
                  Operator Code
                </label>
                <input
                  type="text"
                  placeholder="e.g. OP-02"
                  value={newCode}
                  onChange={(e) => setNewCode(e.target.value)}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus:border-ink"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-muted-foreground mb-1">
                    Role
                  </label>
                  <select
                    value={newRole}
                    onChange={(e) => setNewRole(e.target.value as "operator" | "manager")}
                    className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm outline-none focus:border-ink"
                  >
                    <option value="operator">Operator</option>
                    <option value="manager">Manager</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-muted-foreground mb-1">
                    Daily Goal
                  </label>
                  <input
                    type="number"
                    min={1}
                    required
                    value={newGoal}
                    onChange={(e) => setNewGoal(e.target.value)}
                    className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus:border-ink"
                  />
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="rounded-lg border border-border px-3.5 py-1.5 text-xs font-semibold text-foreground hover:bg-muted cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={addingBusy}
                  className="rounded-lg bg-ink px-4 py-1.5 text-xs font-semibold text-card hover:opacity-90 disabled:opacity-50 cursor-pointer"
                >
                  {addingBusy ? "Adding..." : "Add Operator"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </Page>
  );
}