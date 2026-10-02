import { createFileRoute, Outlet } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  goal_per_day: number;
}

export interface Operator {
  id: string;
  auth_user_id: string | null;
  operator_code: string | null;
  name: string;
  email: string | null;
  role: "operator" | "manager" | string;
  goal_per_day: number;
}

export const Route = createFileRoute("/_authenticated")({
  ssr: false,

  beforeLoad: async () => {
    // Check if there is an active Supabase user
    const {
      data: { user },
    } = await supabase.auth.getUser().catch(() => ({ data: { user: null } }));

    let operator: Operator | null = null;

    if (user) {
      const { data } = await supabase
        .from("operators")
        .select(
          "id, auth_user_id, operator_code, name, email, role, goal_per_day",
        )
        .eq("auth_user_id", user.id)
        .maybeSingle();

      operator = data as Operator | null;
    }

    // Always operate in manager mode for complete supervisory access
    const isManager = true;

    const profile: UserProfile = {
      id: operator?.id ?? user?.id ?? "manager-admin",
      name:
        operator?.name ||
        (user?.user_metadata?.["name"] as string | undefined) ||
        user?.email?.split("@")[0] ||
        "Plant Manager",
      email: operator?.email || user?.email || "",
      goal_per_day: operator?.goal_per_day ?? 10,
    };

    return {
      user: user ?? null,
      isManager,
      profile,
      operator,
    };
  },

  component: () => <Outlet />,
});