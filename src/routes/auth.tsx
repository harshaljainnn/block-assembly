import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Card, inputCls } from "@/components/assembly/ui";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/auth")({
  head: () => ({
    meta: [
      { title: "Sign in — Assembly Monitor" },
      { name: "description", content: "Sign in with your work account to access the assembly monitor." },
      { property: "og:title", content: "Sign in — Assembly Monitor" },
      { property: "og:description", content: "Sign in with your work account to access the assembly monitor." },
    ],
  }),
  component: Auth,
});

function Auth() {
  const nav = useNavigate();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setMsg("");
    if (mode === "signin") {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) setMsg(error.message); else nav({ to: "/" });
    } else {
      const { error } = await supabase.auth.signUp({
        email, password,
        options: { emailRedirectTo: window.location.origin, data: { name } },
      });
      setMsg(error ? error.message : "Account created. Check your email to confirm, then sign in.");
    }
    setBusy(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <Card className="w-[min(400px,92vw)]">
        <div className="text-center">
          <div className="text-xl font-bold">Assembly Monitor</div>
          <h1 className="mt-5 text-[22px] font-bold">{mode === "signin" ? "Sign in" : "Create account"}</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">Access is based on your role (manager or operator).</p>
        </div>
        <form className="mt-6 space-y-2.5" onSubmit={submit}>
          {mode === "signup" && (
            <input className={`${inputCls} w-full`} placeholder="Full name" required value={name} onChange={(e) => setName(e.target.value)} />
          )}
          <input className={`${inputCls} w-full`} type="email" placeholder="Work email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <input className={`${inputCls} w-full`} type="password" placeholder="Password" minLength={6} autoComplete={mode === "signin" ? "current-password" : "new-password"} required value={password} onChange={(e) => setPassword(e.target.value)} />
          <button disabled={busy} className="w-full cursor-pointer rounded-lg bg-ink px-3 py-2 text-xs text-card disabled:opacity-60">
            {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="mt-2 min-h-5 text-center text-sm text-muted-foreground">{msg}</p>
        <button type="button" className="mt-1 w-full cursor-pointer text-center text-xs text-primary" onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setMsg(""); }}>
          {mode === "signin" ? "No account? Create one" : "Already have an account? Sign in"}
        </button>
      </Card>
    </div>
  );
}
