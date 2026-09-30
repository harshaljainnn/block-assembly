import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Card, inputCls } from "@/components/assembly/ui";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — Assembly Monitor" },
      { name: "description", content: "Sign in with your assigned work account to access the assembly monitor." },
      { property: "og:title", content: "Sign in — Assembly Monitor" },
      { property: "og:description", content: "Sign in with your assigned work account to access the assembly monitor." },
    ],
  }),
  component: Login,
});

function Login() {
  const nav = useNavigate();
  const [step, setStep] = useState<"email" | "code">("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [msg, setMsg] = useState("");

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <Card className="w-[min(400px,92vw)]">
        <div className="text-center">
          <div className="text-xl font-bold">Assembly Monitor</div>
          <h1 className="mt-5 text-[22px] font-bold">Sign in</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">Sign in with your assigned work account. Access is based on your role.</p>
        </div>
        <form
          className="mt-6"
          onSubmit={(e) => {
            e.preventDefault();
            if (step === "email") { setStep("code"); setMsg("Code sent. Check your email."); }
            else nav({ to: "/" });
          }}
        >
          <input className={`${inputCls} w-full`} type="email" placeholder="Work email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} />
          {step === "code" && (
            <input className={`${inputCls} mt-2.5 w-full`} inputMode="numeric" autoComplete="one-time-code" placeholder="6-digit code" maxLength={6} value={code} onChange={(e) => setCode(e.target.value)} />
          )}
          <button className="mt-2.5 w-full cursor-pointer rounded-lg bg-ink px-3 py-2 text-xs text-card">{step === "email" ? "Continue" : "Sign in"}</button>
        </form>
        <p className="mt-1.5 min-h-5 text-center text-sm text-muted-foreground">{msg}</p>
      </Card>
    </div>
  );
}
