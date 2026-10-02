import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
CV_API_KEY = os.getenv("CV_API_KEY", "assembly-local-key").strip()

USE_LOCAL_MODE = not (bool(SUPABASE_URL) and bool(SUPABASE_SERVICE_ROLE_KEY))

if USE_LOCAL_MODE:
    print("\n=======================================================")
    print(" [MODE] Running in LOCAL / STANDALONE in-memory mode   ")
    print(" Access Web Dashboard: http://localhost:8000           ")
    print("=======================================================\n")
else:
    print(f"[MODE] Connected to Supabase at {SUPABASE_URL}")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Assembly Performance API",
    description="Backend API and Dashboard for Live Block Assembly Inspection",
    version="1.1.0",
)

# Allow React / Hatchable dashboard to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOCAL IN-MEMORY STORAGE
# ============================================================

LOCAL_OPERATORS = {
    "OP001": {"id": "op_001", "operator_code": "OP001", "name": "Harshal (Station Lead)", "role": "lead"},
    "OP002": {"id": "op_002", "operator_code": "OP002", "name": "Operator 2", "role": "operator"},
}

LOCAL_SESSIONS = {}  # cycle_id -> dict
LOCAL_EVENTS = []    # list of event dicts


# ============================================================
# SUPABASE HELPERS
# ============================================================

SUPABASE_REST_URL = f"{SUPABASE_URL.rstrip('/')}/rest/v1" if SUPABASE_URL else ""


def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def supabase_request(
    method: str,
    table: str,
    params: Optional[dict] = None,
    json_data=None,
):
    """
    Communicate with Supabase REST API.
    """
    if USE_LOCAL_MODE:
        return None

    url = f"{SUPABASE_REST_URL}/{table}"

    response = requests.request(
        method=method,
        url=url,
        headers={
            **supabase_headers(),
            "Prefer": "return=representation",
        },
        params=params,
        json=json_data,
        timeout=10,
    )

    if not response.ok:
        print(f"SUPABASE ERROR [{method} {table}]: {response.status_code} {response.text}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Supabase request failed",
                "status": response.status_code,
                "response": response.text,
            },
        )

    if not response.text:
        return None

    return response.json()


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_cv_api_key(api_key: Optional[str]):
    """
    Verify incoming requests from the CV inspection system.
    In local mode with default key, relaxed verification is permitted.
    """
    if USE_LOCAL_MODE and (not CV_API_KEY or CV_API_KEY == "assembly-local-key"):
        return

    if not api_key or not secrets.compare_digest(api_key, CV_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid CV API key",
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class StartAssemblyRequest(BaseModel):
    operator_id: str = Field(
        ...,
        description="Operator code, e.g. OP001",
    )


class EventRequest(BaseModel):
    cycle_id: str

    state_index: Optional[int] = None
    state_name: Optional[str] = None
    state_title: Optional[str] = None

    status: str = Field(
        ...,
        description="holding, advanced, error or completed",
    )

    confidence: Optional[float] = None
    is_valid: Optional[bool] = None

    diagnostic: Optional[str] = None
    consensus: Optional[str] = None
    error_type: Optional[str] = None

    incoming_object: Optional[str] = None
    incoming_confidence: Optional[float] = None
    incoming_expected: Optional[bool] = None

    detections: Optional[dict] = None
    spatial_checks: Optional[dict] = None
    part_counts: Optional[dict] = None


class EndAssemblyRequest(BaseModel):
    cycle_id: str

    status: str = Field(
        ...,
        description="pass or fail",
    )

    failure_reason: Optional[str] = None

    states_completed: Optional[int] = None
    total_states: int = 9


# ============================================================
# HEALTH & METRICS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "assembly-performance-api",
        "mode": "local" if USE_LOCAL_MODE else "supabase",
        "supabase_configured": not USE_LOCAL_MODE,
    }


# ============================================================
# START ASSEMBLY
# ============================================================

@app.post("/api/assembly/start")
def start_assembly(
    data: StartAssemblyRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        op = LOCAL_OPERATORS.get(data.operator_id)
        if not op:
            op = {
                "id": f"op_{data.operator_id.lower()}",
                "operator_code": data.operator_id,
                "name": f"Operator {data.operator_id}",
                "role": "operator",
            }
            LOCAL_OPERATORS[data.operator_id] = op

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(2).upper()
        cycle_id = f"ASM-{timestamp}-{random_part}"
        session_id = f"sess_{secrets.token_hex(4)}"

        session_data = {
            "id": session_id,
            "cycle_id": cycle_id,
            "operator_id": op["id"],
            "operator_code": op["operator_code"],
            "operator_name": op["name"],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
            "failure_reason": None,
        }
        LOCAL_SESSIONS[cycle_id] = session_data

        return {
            "success": True,
            "message": "Assembly started (Local Mode)",
            "assembly": {
                "id": session_id,
                "cycle_id": cycle_id,
                "operator_id": op["operator_code"],
                "operator_name": op["name"],
                "status": "in_progress",
            },
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    operators = supabase_request(
        "GET",
        "operators",
        params={
            "operator_code": f"eq.{data.operator_id}",
            "select": "id,operator_code,name,role",
            "limit": "1",
        },
    )

    if not operators:
        raise HTTPException(
            status_code=404,
            detail=f"Operator '{data.operator_id}' not found",
        )

    operator = operators[0]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(2).upper()
    cycle_id = f"ASM-{timestamp}-{random_part}"

    assembly = supabase_request(
        "POST",
        "assembly_sessions",
        params={
            "select": (
                "id,"
                "cycle_id,"
                "operator_id,"
                "start_time,"
                "status,"
                "states_completed,"
                "total_states"
            )
        },
        json_data={
            "cycle_id": cycle_id,
            "operator_id": operator["id"],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
        },
    )

    if not assembly:
        raise HTTPException(
            status_code=500,
            detail="Failed to create assembly session",
        )

    return {
        "success": True,
        "message": "Assembly started",
        "assembly": {
            "id": assembly[0]["id"],
            "cycle_id": cycle_id,
            "operator_id": operator["operator_code"],
            "operator_name": operator["name"],
            "status": "in_progress",
        },
    }


# ============================================================
# ASSEMBLY EVENT
# ============================================================

@app.post("/api/assembly/event")
def assembly_event(
    data: EventRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        session = LOCAL_SESSIONS.get(data.cycle_id)
        if not session:
            # Auto-provision session if started spontaneously
            session = {
                "id": f"sess_{secrets.token_hex(4)}",
                "cycle_id": data.cycle_id,
                "operator_id": "op_001",
                "operator_code": "OP001",
                "operator_name": "Harshal (Station Lead)",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "duration_seconds": 0,
                "status": "in_progress",
                "states_completed": 0,
                "total_states": 9,
                "failure_reason": None,
            }
            LOCAL_SESSIONS[data.cycle_id] = session

        event_id = f"evt_{secrets.token_hex(4)}"
        event_record = {
            "id": event_id,
            "assembly_id": session["id"],
            "cycle_id": data.cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_index": data.state_index,
            "state_name": data.state_name,
            "state_title": data.state_title,
            "status": data.status,
            "confidence": data.confidence,
            "is_valid": data.is_valid,
            "diagnostic": data.diagnostic,
            "consensus": data.consensus,
            "error_type": data.error_type,
            "incoming_object": data.incoming_object,
            "incoming_confidence": data.incoming_confidence,
            "incoming_expected": data.incoming_expected,
            "detections": data.detections,
            "spatial_checks": data.spatial_checks,
            "part_counts": data.part_counts,
        }
        LOCAL_EVENTS.append(event_record)

        current_completed = session.get("states_completed") or 0
        new_completed = current_completed
        if data.status in ["advanced", "completed"] and data.state_index is not None:
            new_completed = max(current_completed, data.state_index)
            session["states_completed"] = new_completed

        return {
            "success": True,
            "cycle_id": data.cycle_id,
            "event_id": event_id,
            "states_completed": new_completed,
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": (
                "id,"
                "cycle_id,"
                "states_completed,"
                "total_states,"
                "status"
            ),
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    event_data = {
        "assembly_id": assembly["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state_index": data.state_index,
        "state_name": data.state_name,
        "state_title": data.state_title,
        "status": data.status,
        "confidence": data.confidence,
        "is_valid": data.is_valid,
        "diagnostic": data.diagnostic,
        "consensus": data.consensus,
        "error_type": data.error_type,
        "incoming_object": data.incoming_object,
        "incoming_confidence": data.incoming_confidence,
        "incoming_expected": data.incoming_expected,
        "detections": data.detections,
        "spatial_checks": data.spatial_checks,
        "part_counts": data.part_counts,
    }

    event = supabase_request(
        "POST",
        "assembly_events",
        params={"select": "id,assembly_id,timestamp,status,state_index"},
        json_data=event_data,
    )

    current_completed = assembly.get("states_completed") or 0
    new_completed = current_completed

    if (
        data.status in ["advanced", "completed"]
        and data.state_index is not None
    ):
        new_completed = max(
            current_completed,
            data.state_index,
        )

    if new_completed != current_completed:
        supabase_request(
            "PATCH",
            "assembly_sessions",
            params={
                "id": f"eq.{assembly['id']}",
            },
            json_data={
                "states_completed": new_completed,
            },
        )

    return {
        "success": True,
        "cycle_id": data.cycle_id,
        "event_id": event[0]["id"] if event else None,
        "states_completed": new_completed,
    }


# ============================================================
# END ASSEMBLY
# ============================================================

@app.post("/api/assembly/end")
def end_assembly(
    data: EndAssemblyRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        session = LOCAL_SESSIONS.get(data.cycle_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Assembly '{data.cycle_id}' not found",
            )

        start_time = datetime.fromisoformat(session["start_time"].replace("Z", "+00:00"))
        end_time = datetime.now(timezone.utc)
        duration_seconds = max(0, int((end_time - start_time).total_seconds()))

        status = data.status.lower()
        if status not in ["pass", "fail"]:
            raise HTTPException(status_code=400, detail="status must be 'pass' or 'fail'")

        session["end_time"] = end_time.isoformat()
        session["duration_seconds"] = duration_seconds
        session["status"] = status
        session["states_completed"] = (
            data.states_completed if data.states_completed is not None else session.get("states_completed", 0)
        )
        session["total_states"] = data.total_states
        session["failure_reason"] = data.failure_reason

        return {
            "success": True,
            "assembly": session,
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": (
                "id,"
                "cycle_id,"
                "start_time,"
                "states_completed,"
                "total_states"
            ),
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    start_time = datetime.fromisoformat(
        assembly["start_time"].replace("Z", "+00:00")
    )
    end_time = datetime.now(timezone.utc)

    duration_seconds = max(
        0,
        int((end_time - start_time).total_seconds()),
    )

    states_completed = (
        data.states_completed
        if data.states_completed is not None
        else assembly.get("states_completed", 0)
    )

    status = data.status.lower()
    if status not in ["pass", "fail"]:
        raise HTTPException(
            status_code=400,
            detail="status must be 'pass' or 'fail'",
        )

    supabase_request(
        "PATCH",
        "assembly_sessions",
        params={
            "id": f"eq.{assembly['id']}",
        },
        json_data={
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "status": status,
            "states_completed": states_completed,
            "total_states": data.total_states,
            "failure_reason": data.failure_reason,
        },
    )

    return {
        "success": True,
        "assembly": {
            "id": assembly["id"],
            "cycle_id": data.cycle_id,
            "status": status,
            "duration_seconds": duration_seconds,
            "states_completed": states_completed,
            "total_states": data.total_states,
            "failure_reason": data.failure_reason,
            "end_time": end_time.isoformat(),
        },
    }


# ============================================================
# QUERY ENDPOINTS FOR DASHBOARD
# ============================================================

@app.get("/api/assembly/sessions")
def get_sessions():
    if USE_LOCAL_MODE:
        return list(LOCAL_SESSIONS.values())[::-1]
    return supabase_request("GET", "assembly_sessions", params={"order": "start_time.desc", "limit": "50"})


@app.get("/api/assembly/events")
def get_events(cycle_id: Optional[str] = None):
    if USE_LOCAL_MODE:
        if cycle_id:
            return [e for e in LOCAL_EVENTS if e.get("cycle_id") == cycle_id]
        return LOCAL_EVENTS[-50:][::-1]
    params = {"order": "timestamp.desc", "limit": "50"}
    if cycle_id:
        params["cycle_id"] = f"eq.{cycle_id}"
    return supabase_request("GET", "assembly_events", params=params)


@app.get("/api/assembly/latest")
def get_latest():
    if USE_LOCAL_MODE:
        sessions = list(LOCAL_SESSIONS.values())
        latest_session = sessions[-1] if sessions else None
        latest_event = LOCAL_EVENTS[-1] if LOCAL_EVENTS else None
        passes = sum(1 for s in sessions if s.get("status") == "pass")
        fails = sum(1 for s in sessions if s.get("status") == "fail")
        total = len(sessions)
        pass_rate = round((passes / total) * 100, 1) if total > 0 else 100.0

        return {
            "session": latest_session,
            "latest_event": latest_event,
            "total_cycles": total,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": pass_rate,
        }
    return {"status": "ok", "mode": "supabase"}


# ============================================================
# BUILT-IN REAL-TIME WEB DASHBOARD UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def dashboard_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Assembly Performance Dashboard | CV Edge HUD</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .pulse-live { animation: pulse-fast 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 sticky top-0 z-50">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
          <i class="fa-solid fa-microchip text-xl"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Assembly Performance Dashboard
            <span class="text-xs font-mono uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full">CV Station Live</span>
          </h1>
          <p class="text-xs text-slate-400">Real-time Quality Inspection & Sequence Guidance</p>
        </div>
      </div>
      <div class="flex items-center space-x-4">
        <div class="flex items-center gap-2 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400 pulse-live"></span>
          <span id="bridge-status">BRIDGE ACTIVE</span>
        </div>
        <div class="bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 text-xs font-medium px-3 py-1.5 rounded-lg flex items-center gap-2">
          <i class="fa-solid fa-user-gear"></i>
          <span id="operator-id">OP001 (Harshal)</span>
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 space-y-8">
    <!-- Top Stats Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Active Cycle</span>
          <i class="fa-solid fa-barcode text-slate-500"></i>
        </div>
        <div id="stat-cycle-id" class="text-lg font-mono font-bold text-white truncate">Waiting...</div>
        <div id="stat-cycle-status" class="text-xs text-slate-400 mt-1 flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Idle / Awaiting Start
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Progress</span>
          <i class="fa-solid fa-list-check text-slate-500"></i>
        </div>
        <div class="flex items-baseline gap-2">
          <div id="stat-step-num" class="text-3xl font-bold text-white">0</div>
          <span class="text-slate-500 font-medium">/ 8 steps</span>
        </div>
        <div class="w-full bg-slate-800 h-2 rounded-full mt-3 overflow-hidden">
          <div id="stat-step-bar" class="bg-emerald-500 h-full rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Pass Rate</span>
          <i class="fa-solid fa-chart-line text-slate-500"></i>
        </div>
        <div id="stat-pass-rate" class="text-3xl font-bold text-emerald-400">100%</div>
        <div class="text-xs text-slate-400 mt-1">
          <span id="stat-pass-count">0</span> passed, <span id="stat-fail-count" class="text-rose-400">0</span> failed
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">CV Confidence</span>
          <i class="fa-solid fa-bullseye text-slate-500"></i>
        </div>
        <div id="stat-confidence" class="text-3xl font-bold text-cyan-400">--%</div>
        <div id="stat-latency" class="text-xs text-slate-400 mt-1 font-mono">Edge Latency: < 15ms</div>
      </div>
    </div>

    <!-- Active Step Progress Timeline -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-sm">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-5 flex items-center justify-between">
        <span>Sequential Assembly Checklist</span>
        <span id="current-state-title" class="text-sm font-medium text-emerald-400 font-mono">Step 0: Unstarted</span>
      </h2>

      <div class="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2" id="steps-container">
        <!-- 9 steps injected by JS -->
      </div>
    </div>

    <!-- Diagnostic Banner -->
    <div id="diagnostic-banner" class="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-start gap-4 transition-all">
      <div id="banner-icon" class="p-3 bg-slate-800 text-slate-400 rounded-xl text-xl shrink-0">
        <i class="fa-solid fa-circle-info"></i>
      </div>
      <div>
        <div id="banner-status" class="text-sm font-bold uppercase tracking-wider text-slate-300">Ready for Assembly</div>
        <div id="banner-text" class="text-sm text-slate-400 mt-1">
          Waiting for live computer vision events from <code class="bg-slate-800 px-1.5 py-0.5 rounded text-slate-200">python live_demo.py --dashboard http://localhost:8000</code>.
        </div>
      </div>
    </div>

    <!-- Live Event Feed -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-sm">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-base font-bold text-white flex items-center gap-2">
          <i class="fa-solid fa-stream text-emerald-400"></i> Live Verification Feed
        </h2>
        <span class="text-xs font-mono text-slate-400" id="feed-count">0 events</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="text-xs uppercase bg-slate-950/60 text-slate-400 border-b border-slate-800">
            <tr>
              <th class="py-3 px-4">Time</th>
              <th class="py-3 px-4">Cycle ID</th>
              <th class="py-3 px-4">Step</th>
              <th class="py-3 px-4">Status</th>
              <th class="py-3 px-4">Confidence</th>
              <th class="py-3 px-4">Diagnostic Details</th>
            </tr>
          </thead>
          <tbody id="events-table-body" class="divide-y divide-slate-800 text-slate-300 font-mono text-xs">
            <tr>
              <td colspan="6" class="py-8 text-center text-slate-500 font-sans text-sm">
                No events received yet. Start the CV pipeline to begin inspection stream.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    const STEP_NAMES = [
      "0. Unstarted",
      "1. Green + 1 Blue Base",
      "2. Green + 2 Blue Feet",
      "3. First Red Block",
      "4. First Yellow Block",
      "5. Second Red Stack",
      "6. Second Yellow Block",
      "7. Third Red Block",
      "8. Complete 9-Part Assembly"
    ];

    function renderSteps(activeIdx) {
      const container = document.getElementById("steps-container");
      container.innerHTML = "";
      STEP_NAMES.forEach((name, idx) => {
        let bg = "bg-slate-950/50 border-slate-800 text-slate-500";
        let icon = '<i class="fa-regular fa-circle text-xs"></i>';
        if (idx < activeIdx) {
          bg = "bg-emerald-500/10 border-emerald-500/30 text-emerald-400";
          icon = '<i class="fa-solid fa-check text-xs"></i>';
        } else if (idx === activeIdx) {
          bg = "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-1 ring-amber-500/30";
          icon = '<i class="fa-solid fa-spinner fa-spin text-xs"></i>';
        }
        const div = document.createElement("div");
        div.className = `border rounded-xl p-3 flex flex-col items-center text-center transition-all ${bg}`;
        div.innerHTML = `<div class="mb-1">${icon}</div><div class="text-[11px] font-medium leading-tight">${name}</div>`;
        container.appendChild(div);
      });
    }

    renderSteps(0);

    async function pollDashboard() {
      try {
        const [latestRes, eventsRes] = await Promise.all([
          fetch("/api/assembly/latest"),
          fetch("/api/assembly/events")
        ]);

        if (latestRes.ok && eventsRes.ok) {
          const latestData = await latestRes.json();
          const events = await eventsRes.json();

          const sess = latestData.session;
          const evt = latestData.latest_event;

          if (sess) {
            document.getElementById("stat-cycle-id").innerText = sess.cycle_id;
            const completed = sess.states_completed || 0;
            document.getElementById("stat-step-num").innerText = completed;
            document.getElementById("stat-step-bar").style.width = `${Math.min(100, (completed / 8) * 100)}%`;
            renderSteps(completed);

            let statusText = sess.status.toUpperCase();
            document.getElementById("stat-cycle-status").innerHTML =
              `<span class="w-1.5 h-1.5 rounded-full ${sess.status === 'pass' ? 'bg-emerald-400' : 'bg-amber-400'}"></span> ${statusText}`;
            document.getElementById("current-state-title").innerText = STEP_NAMES[completed] || `Step ${completed}`;
          }

          document.getElementById("stat-pass-rate").innerText = `${latestData.pass_rate}%`;
          document.getElementById("stat-pass-count").innerText = latestData.pass_count;
          document.getElementById("stat-fail-count").innerText = latestData.fail_count;

          if (evt) {
            const confPct = evt.confidence ? `${Math.round(evt.confidence * 100)}%` : '--%';
            document.getElementById("stat-confidence").innerText = confPct;

            const banner = document.getElementById("diagnostic-banner");
            const icon = document.getElementById("banner-icon");
            const status = document.getElementById("banner-status");
            const text = document.getElementById("banner-text");

            if (evt.status === "error" || (evt.diagnostic && evt.diagnostic.includes("INCORRECT"))) {
              banner.className = "bg-rose-950/30 border border-rose-500/40 rounded-2xl p-5 flex items-start gap-4";
              icon.className = "p-3 bg-rose-500/20 text-rose-400 rounded-xl text-xl shrink-0";
              icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
              status.className = "text-sm font-bold uppercase tracking-wider text-rose-400";
              status.innerText = "DEFECT / MISPLACEMENT DETECTED";
              text.innerText = evt.diagnostic || "Component placed incorrectly. Please correct assembly.";
            } else if (evt.status === "completed" || evt.status === "advanced") {
              banner.className = "bg-emerald-950/30 border border-emerald-500/40 rounded-2xl p-5 flex items-start gap-4";
              icon.className = "p-3 bg-emerald-500/20 text-emerald-400 rounded-xl text-xl shrink-0";
              icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
              status.className = "text-sm font-bold uppercase tracking-wider text-emerald-400";
              status.innerText = `ADVANCED TO ${STEP_NAMES[evt.state_index] || evt.state_name}`;
              text.innerText = evt.diagnostic || "Part verified successfully with high spatial consensus.";
            } else if (evt.status === "holding") {
              banner.className = "bg-slate-900 border border-cyan-500/30 rounded-2xl p-5 flex items-start gap-4";
              icon.className = "p-3 bg-cyan-500/10 text-cyan-400 rounded-xl text-xl shrink-0";
              icon.innerHTML = '<i class="fa-solid fa-eye pulse-live"></i>';
              status.className = "text-sm font-bold uppercase tracking-wider text-cyan-300";
              status.innerText = `ACTIVE: ${STEP_NAMES[evt.state_index] || evt.state_name}`;
              text.innerText = evt.diagnostic || "Inspecting workspace and verifying assembly sequence...";
            }
          }

          if (Array.isArray(events) && events.length > 0) {
            document.getElementById("feed-count").innerText = `${events.length} events logged`;
            const tbody = document.getElementById("events-table-body");
            tbody.innerHTML = events.slice(0, 15).map(e => {
              const time = new Date(e.timestamp).toLocaleTimeString();
              let badge = "bg-slate-800 text-slate-300";
              if (e.status === "completed") badge = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
              else if (e.status === "advanced") badge = "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30";
              else if (e.status === "error") badge = "bg-rose-500/20 text-rose-400 border border-rose-500/30";
              else if (e.status === "holding") badge = "bg-slate-800 text-cyan-400 border border-slate-700";

              return `<tr>
                <td class="py-3 px-4 text-slate-400">${time}</td>
                <td class="py-3 px-4 text-indigo-300">${e.cycle_id || '--'}</td>
                <td class="py-3 px-4 font-semibold text-white">${e.state_index !== undefined ? (STEP_NAMES[e.state_index] || 'Step ' + e.state_index) : (e.state_name || '--')}</td>
                <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] uppercase font-bold ${badge}">${e.status}</span></td>
                <td class="py-3 px-4 text-cyan-400">${e.confidence ? Math.round(e.confidence * 100) + '%' : '--'}</td>
                <td class="py-3 px-4 text-slate-300 font-sans text-xs truncate max-w-xs">${e.diagnostic || '--'}</td>
              </tr>`;
            }).join("");
          }
        }
      } catch (err) {
        console.error("Dashboard poll error:", err);
      }
    }

    setInterval(pollDashboard, 1000);
    pollDashboard();
  </script>
</body>
</html>"""