export const liveEvents = [
  { time: "10:32:04", state: "State 4", object: "Red Block", confidence: 0.98, event: "Correct placement", status: "SUCCESS" },
  { time: "10:32:18", state: "State 5", object: "Blue Block", confidence: 0.91, event: "Correct placement", status: "SUCCESS" },
  { time: "10:32:41", state: "State 6", object: "Yellow Block", confidence: 0.87, event: "Wrong sequence — expected Red Block", status: "FAILED" },
];

export type Operator = { user_id: string; name: string; email: string; goal_per_day: number; completed: number; pass_rate: number };
export const demoOperators: Operator[] = [
  { user_id: "demo-01", name: "Operator 01", email: "operator01@demo.local", goal_per_day: 20, completed: 14, pass_rate: 96 },
  { user_id: "demo-02", name: "Operator 02", email: "operator02@demo.local", goal_per_day: 18, completed: 12, pass_rate: 92 },
  { user_id: "demo-03", name: "Operator 03", email: "operator03@demo.local", goal_per_day: 15, completed: 10, pass_rate: 90 },
  { user_id: "demo-04", name: "Operator 04", email: "operator04@demo.local", goal_per_day: 12, completed: 6, pass_rate: 91 },
];

export type Cycle = { id: string; start_time: string; end_time: string; duration: number; state: string; status: "Complete" | "Failed" | "In Progress"; failure_reason: string };
export const historyRows: Cycle[] = [
  { id: "ASM-00124", start_time: "10:30:01", end_time: "10:34:42", duration: 281, state: "10/10", status: "Complete", failure_reason: "" },
  { id: "ASM-00123", start_time: "10:22:10", end_time: "10:26:38", duration: 268, state: "8/10", status: "Failed", failure_reason: "Wrong sequence" },
  { id: "ASM-00122", start_time: "10:14:03", end_time: "10:18:21", duration: 258, state: "10/10", status: "Complete", failure_reason: "" },
];

export const analyticsRows = [
  { status: "Pass", duration: 281 }, { status: "Pass", duration: 268 }, { status: "Failed", duration: 290, failure_reason: "Wrong sequence" },
  { status: "Pass", duration: 244 }, { status: "Pass", duration: 252 }, { status: "Pass", duration: 231 },
  { status: "Failed", duration: 310, failure_reason: "Missed object" }, { status: "Pass", duration: 220 },
  { status: "Pass", duration: 248 }, { status: "Pass", duration: 237 },
];

export const fmtDuration = (sec: number) => `${Math.floor(sec / 60)}m ${String(Math.round(sec) % 60).padStart(2, "0")}s`;
