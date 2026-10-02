export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type Database = {
  __InternalSupabase: {
    PostgrestVersion: "14.18";
  };

  public: {
    Tables: {
      assembly_events: {
        Row: {
          assembly_id: string;
          confidence: number | null;
          consensus: string | null;
          detections: Json | null;
          diagnostic: string | null;
          error_type: string | null;
          id: string;
          incoming_confidence: number | null;
          incoming_expected: boolean | null;
          incoming_object: string | null;
          is_valid: boolean | null;
          part_counts: Json | null;
          spatial_checks: Json | null;
          state_index: number | null;
          state_name: string | null;
          state_title: string | null;
          status: string;
          timestamp: string;
        };

        Insert: {
          assembly_id: string;
          confidence?: number | null;
          consensus?: string | null;
          detections?: Json | null;
          diagnostic?: string | null;
          error_type?: string | null;
          id?: string;
          incoming_confidence?: number | null;
          incoming_expected?: boolean | null;
          incoming_object?: string | null;
          is_valid?: boolean | null;
          part_counts?: Json | null;
          spatial_checks?: Json | null;
          state_index?: number | null;
          state_name?: string | null;
          state_title?: string | null;
          status: string;
          timestamp?: string;
        };

        Update: {
          assembly_id?: string;
          confidence?: number | null;
          consensus?: string | null;
          detections?: Json | null;
          diagnostic?: string | null;
          error_type?: string | null;
          id?: string;
          incoming_confidence?: number | null;
          incoming_expected?: boolean | null;
          incoming_object?: string | null;
          is_valid?: boolean | null;
          part_counts?: Json | null;
          spatial_checks?: Json | null;
          state_index?: number | null;
          state_name?: string | null;
          state_title?: string | null;
          status?: string;
          timestamp?: string;
        };

        Relationships: [
          {
            foreignKeyName: "assembly_events_assembly_id_fkey";
            columns: ["assembly_id"];
            isOneToOne: false;
            referencedRelation: "assembly_sessions";
            referencedColumns: ["id"];
          },
        ];
      };

      assembly_sessions: {
        Row: {
          created_at: string;
          cycle_id: string;
          duration_seconds: number | null;
          end_time: string | null;
          failure_reason: string | null;
          id: string;
          operator_id: string | null;
          start_time: string;
          states_completed: number;
          status: string;
          total_states: number;
        };

        Insert: {
          created_at?: string;
          cycle_id: string;
          duration_seconds?: number | null;
          end_time?: string | null;
          failure_reason?: string | null;
          id?: string;
          operator_id?: string | null;
          start_time?: string;
          states_completed?: number;
          status?: string;
          total_states?: number;
        };

        Update: {
          created_at?: string;
          cycle_id?: string;
          duration_seconds?: number | null;
          end_time?: string | null;
          failure_reason?: string | null;
          id?: string;
          operator_id?: string | null;
          start_time?: string;
          states_completed?: number;
          status?: string;
          total_states?: number;
        };

        Relationships: [
          {
            foreignKeyName: "assembly_sessions_operator_id_fkey";
            columns: ["operator_id"];
            isOneToOne: false;
            referencedRelation: "operators";
            referencedColumns: ["id"];
          },
        ];
      };

      operator_sessions: {
        Row: {
          id: string;
          login_time: string;
          logout_time: string | null;
          operator_id: string;
        };

        Insert: {
          id?: string;
          login_time?: string;
          logout_time?: string | null;
          operator_id: string;
        };

        Update: {
          id?: string;
          login_time?: string;
          logout_time?: string | null;
          operator_id?: string;
        };

        Relationships: [
          {
            foreignKeyName: "operator_sessions_operator_id_fkey";
            columns: ["operator_id"];
            isOneToOne: false;
            referencedRelation: "operators";
            referencedColumns: ["id"];
          },
        ];
      };

      operators: {
        Row: {
          auth_user_id: string | null;
          created_at: string;
          email: string | null;
          goal_per_day: number;
          id: string;
          name: string;
          operator_code: string | null;
          role: string;
        };

        Insert: {
          auth_user_id?: string | null;
          created_at?: string;
          email?: string | null;
          goal_per_day?: number;
          id?: string;
          name: string;
          operator_code?: string | null;
          role?: string;
        };

        Update: {
          auth_user_id?: string | null;
          created_at?: string;
          email?: string | null;
          goal_per_day?: number;
          id?: string;
          name?: string;
          operator_code?: string | null;
          role?: string;
        };

        Relationships: [];
      };
    };

    Views: {
      [_ in never]: never;
    };

    Functions: {
      [_ in never]: never;
    };

    Enums: {
      [_ in never]: never;
    };

    CompositeTypes: {
      [_ in never]: never;
    };
  };
};

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">;

type DefaultSchema =
  DatabaseWithoutInternals[Extract<keyof Database, "public">];

export type Tables<
  DefaultSchemaTableNameOrOptions extends
  | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
  | { schema: keyof DatabaseWithoutInternals },
  TableName extends (
    DefaultSchemaTableNameOrOptions extends {
      schema: keyof DatabaseWithoutInternals;
    }
    ? keyof (
      DatabaseWithoutInternals[
      DefaultSchemaTableNameOrOptions["schema"]
      ]["Tables"] &
      DatabaseWithoutInternals[
      DefaultSchemaTableNameOrOptions["schema"]
      ]["Views"]
    )
    : never
  ) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? (
    DatabaseWithoutInternals[
    DefaultSchemaTableNameOrOptions["schema"]
    ]["Tables"] &
    DatabaseWithoutInternals[
    DefaultSchemaTableNameOrOptions["schema"]
    ]["Views"]
  )[TableName] extends {
    Row: infer R;
  }
  ? R
  : never
  : DefaultSchemaTableNameOrOptions extends keyof (
    DefaultSchema["Tables"] & DefaultSchema["Views"]
  )
  ? (
    DefaultSchema["Tables"] & DefaultSchema["Views"]
  )[DefaultSchemaTableNameOrOptions] extends {
    Row: infer R;
  }
  ? R
  : never
  : never;

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
  | keyof DefaultSchema["Tables"]
  | { schema: keyof DatabaseWithoutInternals },
  TableName extends (
    DefaultSchemaTableNameOrOptions extends {
      schema: keyof DatabaseWithoutInternals;
    }
    ? keyof DatabaseWithoutInternals[
    DefaultSchemaTableNameOrOptions["schema"]
    ]["Tables"]
    : never
  ) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[
  DefaultSchemaTableNameOrOptions["schema"]
  ]["Tables"][TableName] extends {
    Insert: infer I;
  }
  ? I
  : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DefaultSchema["Tables"][
  DefaultSchemaTableNameOrOptions
  ] extends {
    Insert: infer I;
  }
  ? I
  : never
  : never;

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
  | keyof DefaultSchema["Tables"]
  | { schema: keyof DatabaseWithoutInternals },
  TableName extends (
    DefaultSchemaTableNameOrOptions extends {
      schema: keyof DatabaseWithoutInternals;
    }
    ? keyof DatabaseWithoutInternals[
    DefaultSchemaTableNameOrOptions["schema"]
    ]["Tables"]
    : never
  ) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[
  DefaultSchemaTableNameOrOptions["schema"]
  ]["Tables"][TableName] extends {
    Update: infer U;
  }
  ? U
  : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DatabaseWithoutInternals[
  "public"
  ]["Tables"][DefaultSchemaTableNameOrOptions] extends {
    Update: infer U;
  }
  ? U
  : never
  : never;

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
  | keyof DefaultSchema["Enums"]
  | { schema: keyof DatabaseWithoutInternals },
  EnumName extends (
    DefaultSchemaEnumNameOrOptions extends {
      schema: keyof DatabaseWithoutInternals;
    }
    ? keyof DatabaseWithoutInternals[
    DefaultSchemaEnumNameOrOptions["schema"]
    ]["Enums"]
    : never
  ) = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[
  DefaultSchemaEnumNameOrOptions["schema"]
  ]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
  ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
  : never;

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
  | keyof DefaultSchema["CompositeTypes"]
  | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends (
    PublicCompositeTypeNameOrOptions extends {
      schema: keyof DatabaseWithoutInternals;
    }
    ? keyof DatabaseWithoutInternals[
    PublicCompositeTypeNameOrOptions["schema"]
    ]["CompositeTypes"]
    : never
  ) = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[
  PublicCompositeTypeNameOrOptions["schema"]
  ]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema[
  "CompositeTypes"
  ]
  ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
  : never;

export const Constants = {
  public: {
    Enums: {},
  },
} as const;