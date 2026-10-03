import os
import secrets
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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
DROIDCAM_DEFAULT_URL = os.getenv("DROIDCAM_URL", "http://192.168.2.217:4747/video").strip()

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
    version="1.3.0",
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
# LOCAL IN-MEMORY STORAGE & FRAME BUFFER
# ============================================================

LOCAL_OPERATORS = {
    "OP001": {"id": "op_001", "operator_code": "OP001", "name": "Harshal (Station Lead)", "role": "lead"},
    "OP002": {"id": "op_002", "operator_code": "OP002", "name": "Operator 2", "role": "operator"},
}

LOCAL_SESSIONS = {}  # cycle_id -> dict
LOCAL_EVENTS = []    # list of event dicts

LATEST_HUD_FRAME: Optional[bytes] = None
LATEST_FRAME_TIME: float = 0.0
LATEST_ACTIVE_CYCLE_ID: Optional[str] = None
DAILY_GOAL: int = 50


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


class GoalRequest(BaseModel):
    goal: int = Field(50, ge=1, le=1000)


class ResetAssemblyRequest(BaseModel):
    operator_id: Optional[str] = "OP001"


# ============================================================
# HEALTH & CAMERA STREAM PROXIES
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "assembly-performance-api",
        "mode": "local" if USE_LOCAL_MODE else "supabase",
        "supabase_configured": not USE_LOCAL_MODE,
        "hud_stream_active": bool(LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0)),
    }


@app.post("/api/camera/frame")
async def receive_frame(
    request: Request,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    """
    Receives compressed JPEG HUD frames from live_demo.py with all YOLO
    bounding boxes, labels, and PiP insets.
    """
    global LATEST_HUD_FRAME, LATEST_FRAME_TIME
    verify_cv_api_key(x_cv_api_key)
    LATEST_HUD_FRAME = await request.body()
    LATEST_FRAME_TIME = time.time()
    return {"status": "ok"}


@app.get("/api/camera/hud_stream")
def hud_stream():
    """
    Streams the live AI-annotated HUD frame (with bounding boxes, PiP, checklist)
    to browser clients as a standard MJPEG stream.
    """
    def generate():
        last_sent_time = 0.0
        while True:
            if LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0):
                # Yield when a new frame arrives or periodic heartbeat
                if LATEST_FRAME_TIME != last_sent_time or (time.time() - last_sent_time > 1.0):
                    last_sent_time = LATEST_FRAME_TIME
                    frame_data = LATEST_HUD_FRAME
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                        + frame_data
                        + b"\r\n"
                    )
            time.sleep(0.02)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.get("/api/camera/stream")
def camera_stream():
    """
    Proxies the DroidCam raw MJPEG stream directly to browser clients.
    """
    def generate():
        try:
            req = urllib.request.urlopen(DROIDCAM_DEFAULT_URL, timeout=3.0)
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                yield chunk
        except Exception:
            return

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=--dcmjpeg",
    )


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
        cycle_number = len(LOCAL_SESSIONS) + 1

        session_data = {
            "id": session_id,
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
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
                "cycle_number": cycle_number,
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
        global LATEST_ACTIVE_CYCLE_ID
        LATEST_ACTIVE_CYCLE_ID = data.cycle_id

        session = LOCAL_SESSIONS.get(data.cycle_id)
        if not session:
            cycle_number = len(LOCAL_SESSIONS) + 1
            session = {
                "id": f"sess_{secrets.token_hex(4)}",
                "cycle_id": data.cycle_id,
                "cycle_number": cycle_number,
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

        if data.state_index is not None:
            session["states_completed"] = data.state_index
            session["current_state_index"] = data.state_index
        else:
            session["states_completed"] = session.get("states_completed", 0)

        return {
            "success": True,
            "cycle_id": data.cycle_id,
            "event_id": event_id,
            "states_completed": session["states_completed"],
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
        data.status in ["advanced", "completed", "holding"]
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
# GOAL & RESET ENDPOINTS
# ============================================================

@app.post("/api/assembly/goal")
def set_goal(data: GoalRequest):
    global DAILY_GOAL
    DAILY_GOAL = data.goal
    return {"success": True, "goal_target": DAILY_GOAL}


@app.post("/api/assembly/reset")
def reset_assembly(
    data: Optional[ResetAssemblyRequest] = None,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    if USE_LOCAL_MODE:
        op_code = (data.operator_id if data else None) or "OP001"
        op = LOCAL_OPERATORS.get(op_code, {"id": "op_001", "operator_code": "OP001", "name": "Harshal (Station Lead)", "role": "lead"})
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(2).upper()
        cycle_id = f"ASM-{timestamp}-{random_part}"
        session_id = f"sess_{secrets.token_hex(4)}"
        cycle_number = len(LOCAL_SESSIONS) + 1

        session_data = {
            "id": session_id,
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
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

        event_id = f"evt_{secrets.token_hex(4)}"
        event_record = {
            "id": event_id,
            "assembly_id": session_id,
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_index": 0,
            "state_name": "state_0_unstarted",
            "state_title": "0. Unstarted",
            "status": "holding",
            "confidence": 1.0,
            "is_valid": True,
            "diagnostic": "Station Ready for Next Unit",
            "consensus": "state_0_unstarted",
            "error_type": None,
            "incoming_object": None,
            "incoming_confidence": None,
            "incoming_expected": None,
            "detections": None,
            "spatial_checks": None,
            "part_counts": None,
        }
        LOCAL_EVENTS.append(event_record)

        return {
            "success": True,
            "message": "Assembly reset to Step 0 (Local Mode)",
            "assembly": {
                "id": session_id,
                "cycle_id": cycle_id,
                "cycle_number": cycle_number,
                "operator_id": op["operator_code"],
                "operator_name": op["name"],
                "status": "in_progress",
                "states_completed": 0,
            },
        }

    return {"success": True, "message": "Reset called"}


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
        latest_event = LOCAL_EVENTS[-1] if LOCAL_EVENTS else None

        active_cid = None
        if latest_event and latest_event.get("cycle_id") in LOCAL_SESSIONS:
            active_cid = latest_event["cycle_id"]
        elif LATEST_ACTIVE_CYCLE_ID and LATEST_ACTIVE_CYCLE_ID in LOCAL_SESSIONS:
            active_cid = LATEST_ACTIVE_CYCLE_ID

        if active_cid and active_cid in LOCAL_SESSIONS:
            latest_session = LOCAL_SESSIONS[active_cid]
        elif sessions:
            latest_session = sessions[-1]
        else:
            latest_session = None

        current_step = 0
        if latest_event and latest_event.get("state_index") is not None:
            current_step = latest_event["state_index"]
        elif latest_session and latest_session.get("states_completed") is not None:
            current_step = latest_session["states_completed"]

        passes = sum(1 for s in sessions if s.get("status") == "pass")
        fails = sum(1 for s in sessions if s.get("status") == "fail")
        total = len(sessions)
        pass_rate = round((passes / total) * 100, 1) if total > 0 else 100.0

        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sessions_today = [
            s for s in sessions
            if (s.get("start_time") or "").startswith(today_prefix)
        ]
        cycles_done_today = sum(
            1 for s in sessions_today
            if s.get("status") in ["pass", "fail", "completed"]
        )
        today_passes = sum(1 for s in sessions_today if s.get("status") == "pass")
        today_fails = sum(1 for s in sessions_today if s.get("status") == "fail")

        active_cycle_number = latest_session.get("cycle_number") if latest_session else (len(sessions) if sessions else 1)
        hud_active = bool(LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0))

        return {
            "session": latest_session,
            "latest_event": latest_event,
            "current_step": current_step,
            "total_cycles": total,
            "active_cycle_number": active_cycle_number,
            "cycles_done_today": cycles_done_today,
            "goal_target": DAILY_GOAL,
            "today_passes": today_passes,
            "today_fails": today_fails,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": pass_rate,
            "hud_active": hud_active,
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
  <title>Assembly Performance Dashboard | CV Edge Station</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    .pulse-live { animation: pulse-fast 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
  <!-- Top Navigation Bar -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 py-4 sticky top-0 z-50">
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
      <div class="flex items-center space-x-3">
        <div class="flex items-center gap-2 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400 pulse-live"></span>
          <span id="bridge-status">BRIDGE ACTIVE</span>
        </div>
        <div class="bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 text-xs font-medium px-3 py-1.5 rounded-lg flex items-center gap-2 font-mono">
          <span class="text-indigo-400 font-bold">CYCLE <span id="header-cycle-num">#1</span></span>
        </div>
        <button onclick="triggerResetCycle()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-emerald-500/50 rounded-lg text-xs font-medium flex items-center gap-1.5 transition shadow-sm" title="Reset checklist back to Step 0">
          <i class="fa-solid fa-arrows-rotate text-emerald-400"></i>
          <span>Reset / Next Cycle</span>
        </button>
        <div class="bg-slate-800/80 border border-slate-700 text-slate-300 text-xs font-medium px-3 py-1.5 rounded-lg flex items-center gap-2">
          <i class="fa-solid fa-user-gear text-indigo-400"></i>
          <span id="operator-id">OP001 (Harshal)</span>
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 space-y-8">
    
    <!-- 1. LIVE AI COMPUTER VISION HUD STREAM SECTION -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      <div class="px-6 py-4 border-b border-slate-800 bg-slate-900/90 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3">
          <div class="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
            <i class="fa-solid fa-brain text-base"></i>
          </div>
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              Live AI Inspection Stream
              <span id="cam-status-pill" class="text-[11px] font-mono uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded-full flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-live"></span>
                <span id="cam-status-text">AI BOUNDING BOXES ACTIVE</span>
              </span>
            </h2>
            <p class="text-xs text-slate-400">Stream: <span id="cam-active-mode" class="font-mono text-emerald-300">YOLO Detection & Sequential HUD</span></p>
          </div>
        </div>

        <div class="flex items-center gap-3">
          <!-- Feed Switcher (HUD vs Raw Phone) -->
          <div class="inline-flex rounded-xl bg-slate-950 p-1 border border-slate-800 text-xs">
            <button id="btn-mode-hud" onclick="switchStreamMode('hud')" class="px-3 py-1.5 rounded-lg font-medium transition bg-emerald-600 text-white flex items-center gap-1.5">
              <i class="fa-solid fa-cube text-xs"></i> AI HUD Feed (Boxes)
            </button>
            <button id="btn-mode-raw" onclick="switchStreamMode('raw')" class="px-3 py-1.5 rounded-lg font-medium transition text-slate-400 hover:text-white flex items-center gap-1.5">
              <i class="fa-solid fa-mobile-screen text-xs"></i> Raw Phone
            </button>
          </div>

          <div id="raw-url-box" class="hidden flex items-center bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-300 gap-2">
            <i class="fa-solid fa-link text-slate-500"></i>
            <input id="stream-url-input" type="text" value="http://192.168.2.217:4747/video" class="bg-transparent text-xs font-mono text-slate-200 outline-none w-48 placeholder-slate-600" />
            <button onclick="updateCameraStream()" class="hover:text-emerald-400 px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 transition" title="Apply URL">
              <i class="fa-solid fa-arrow-rotate-right"></i>
            </button>
          </div>

          <button onclick="toggleCamFullscreen()" class="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl transition text-xs" title="Expand View">
            <i class="fa-solid fa-expand"></i>
          </button>
        </div>
      </div>

      <div class="relative bg-slate-950 flex items-center justify-center min-h-[380px] max-h-[540px] overflow-hidden group" id="cam-wrapper">
        <img id="camera-stream-img" src="/api/camera/hud_stream" alt="AI Inspection HUD Stream" class="w-full h-auto max-h-[520px] object-contain transition-all" onload="onStreamLoaded()" onerror="onStreamError()" />
        
        <!-- Stream Disconnected Overlay -->
        <div id="cam-fallback" class="hidden absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center">
          <div class="p-6 bg-slate-900 border border-slate-800 rounded-2xl max-w-md shadow-2xl space-y-3">
            <div class="w-12 h-12 bg-amber-500/10 text-amber-400 rounded-2xl flex items-center justify-center mx-auto text-xl border border-amber-500/20">
              <i class="fa-solid fa-microchip"></i>
            </div>
            <h3 class="text-sm font-bold text-white">AI HUD Stream Waiting for Detector</h3>
            <p class="text-xs text-slate-400">Launch the live detector in your terminal to start streaming the real-time AI feed with bounding boxes:</p>
            <div class="bg-slate-950 border border-slate-800 p-2.5 rounded-xl font-mono text-[11px] text-emerald-400 select-all">
              python live_demo.py --camera phone --dashboard http://localhost:8000
            </div>
            <div class="pt-2 flex items-center justify-center gap-3">
              <button onclick="switchStreamMode('raw')" class="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold transition">
                View Raw Phone Feed
              </button>
              <button onclick="switchStreamMode('hud')" class="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold transition">
                Retry AI Feed
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 2. TOP STATS ROW (5 CARDS) -->
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      
      <!-- Card 1: Daily Goal Target -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Daily Goal</span>
          <i class="fa-solid fa-bullseye text-indigo-400 text-base"></i>
        </div>
        <div class="flex items-baseline gap-2">
          <div id="stat-goal-val" class="text-3xl font-bold text-indigo-400">50</div>
          <span class="text-xs text-slate-500 font-medium">units</span>
        </div>
        <div class="text-[11px] text-slate-400 mt-3 flex items-center justify-between">
          <span>Shift Target</span>
          <span class="text-indigo-300 font-mono text-[10px] bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">Fixed 50</span>
        </div>
      </div>

      <!-- Card 2: Cycles Done Today (number / fixed goal) -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Cycles Done Today</span>
          <i class="fa-solid fa-calendar-check text-emerald-400 text-base"></i>
        </div>
        <div class="flex items-baseline gap-2">
          <span id="stat-cycles-done-today" class="text-3xl font-bold text-white">0</span>
          <span class="text-slate-500 text-2xl font-light">/</span>
          <span id="stat-goal-target-display" class="text-2xl font-bold text-indigo-400">50</span>
        </div>
        <div class="w-full bg-slate-800 h-2 rounded-full mt-3 overflow-hidden">
          <div id="stat-goal-bar" class="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
        </div>
        <div class="text-xs text-slate-400 mt-2 flex items-center justify-between">
          <span><span id="stat-today-passed" class="text-emerald-400 font-medium">0</span> passed, <span id="stat-today-failed" class="text-rose-400 font-medium">0</span> failed</span>
          <span id="stat-goal-pct" class="text-emerald-400 font-mono font-semibold">0%</span>
        </div>
      </div>

      <!-- Card 3: Assembly Progress -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Progress</span>
          <i class="fa-solid fa-list-check text-slate-500 text-base"></i>
        </div>
        <div class="flex items-baseline gap-2">
          <div id="stat-step-num" class="text-3xl font-bold text-white">0</div>
          <span class="text-slate-500 font-medium">/ 8 steps</span>
        </div>
        <div class="w-full bg-slate-800 h-2 rounded-full mt-3 overflow-hidden">
          <div id="stat-step-bar" class="bg-emerald-500 h-full rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
        <div class="text-xs text-slate-400 mt-2 truncate">
          <span id="stat-cycle-status-text" class="text-slate-300 font-medium">Ready</span>
        </div>
      </div>

      <!-- Card 4: Quality Pass Rate -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">Pass Rate</span>
          <i class="fa-solid fa-chart-line text-emerald-400"></i>
        </div>
        <div id="stat-pass-rate" class="text-3xl font-bold text-emerald-400">100%</div>
        <div class="text-xs text-slate-400 mt-1">
          Total: <span id="stat-total-cycles" class="text-slate-300 font-medium">0</span> sessions
        </div>
      </div>

      <!-- Card 5: AI Confidence -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm col-span-2 md:col-span-1">
        <div class="flex items-center justify-between text-slate-400 mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider">AI Confidence</span>
          <i class="fa-solid fa-bullseye text-cyan-400"></i>
        </div>
        <div id="stat-confidence" class="text-3xl font-bold text-cyan-400">--%</div>
        <div id="stat-latency" class="text-xs text-slate-400 mt-1 font-mono">Edge Latency: < 15ms</div>
      </div>
    </div>

    <!-- 3. SEQUENTIAL ASSEMBLY CHECKLIST (GREEN TICKS & UPCOMING SPINNER) -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-sm">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Sequential Assembly Checklist</h2>
          <p class="text-xs text-slate-500">Green tick = Verified State &nbsp;•&nbsp; Spinner = Upcoming State to Assemble</p>
        </div>
        <div class="flex items-center gap-2">
          <span id="current-state-title" class="text-sm font-medium text-emerald-400 font-mono bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-lg">
            0. Unstarted
          </span>
          <button onclick="triggerResetCycle()" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs font-mono transition flex items-center gap-1.5" title="Manual reset">
            <i class="fa-solid fa-arrow-rotate-left text-slate-400"></i> Reset
          </button>
        </div>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-2.5" id="steps-container">
        <!-- 9 steps dynamically injected -->
      </div>
    </div>

    <!-- 4. DIAGNOSTIC ALERT BANNER -->
    <div id="diagnostic-banner" class="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-start gap-4 transition-all">
      <div id="banner-icon" class="p-3 bg-slate-800 text-slate-400 rounded-xl text-xl shrink-0">
        <i class="fa-solid fa-circle-info"></i>
      </div>
      <div>
        <div id="banner-status" class="text-sm font-bold uppercase tracking-wider text-slate-300">Ready for Assembly</div>
        <div id="banner-text" class="text-sm text-slate-400 mt-1">
          Waiting for live computer vision events from <code class="bg-slate-800 px-1.5 py-0.5 rounded text-slate-200">python live_demo.py --camera phone --dashboard http://localhost:8000</code>.
        </div>
      </div>
    </div>

    <!-- 5. LIVE VERIFICATION FEED TABLE -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-sm">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-base font-bold text-white flex items-center gap-2">
          <i class="fa-solid fa-stream text-emerald-400"></i> Live Verification Feed
        </h2>
        <span class="text-xs font-mono text-slate-400" id="feed-count">0 events logged</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="text-xs uppercase bg-slate-950/60 text-slate-400 border-b border-slate-800">
            <tr>
              <th class="py-3 px-4">Time</th>
              <th class="py-3 px-4">Cycle ID</th>
              <th class="py-3 px-4">Step Title</th>
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

    // ========================================================
    // CAMERA STREAM CONTROLS (AI HUD vs RAW DROIDCAM)
    // ========================================================
    let currentStreamMode = "hud"; // "hud" (AI with bounding boxes) or "raw"
    const savedStreamUrl = localStorage.getItem("droidcam_stream_url") || "http://192.168.2.217:4747/video";
    const streamInput = document.getElementById("stream-url-input");
    const streamImg = document.getElementById("camera-stream-img");
    const fallbackBox = document.getElementById("cam-fallback");
    const camStatusPill = document.getElementById("cam-status-pill");
    const camStatusText = document.getElementById("cam-status-text");
    const camActiveMode = document.getElementById("cam-active-mode");
    const rawBox = document.getElementById("raw-url-box");

    streamInput.value = savedStreamUrl;

    function switchStreamMode(mode) {
      currentStreamMode = mode;
      const btnHud = document.getElementById("btn-mode-hud");
      const btnRaw = document.getElementById("btn-mode-raw");

      fallbackBox.classList.add("hidden");

      if (mode === "hud") {
        btnHud.className = "px-3 py-1.5 rounded-lg font-medium transition bg-emerald-600 text-white flex items-center gap-1.5";
        btnRaw.className = "px-3 py-1.5 rounded-lg font-medium transition text-slate-400 hover:text-white flex items-center gap-1.5";
        rawBox.classList.add("hidden");
        camActiveMode.innerText = "YOLO Detection & Sequential HUD";
        streamImg.src = "/api/camera/hud_stream?t=" + Date.now();
        camStatusText.innerText = "AI BOUNDING BOXES ACTIVE";
      } else {
        btnRaw.className = "px-3 py-1.5 rounded-lg font-medium transition bg-indigo-600 text-white flex items-center gap-1.5";
        btnHud.className = "px-3 py-1.5 rounded-lg font-medium transition text-slate-400 hover:text-white flex items-center gap-1.5";
        rawBox.classList.remove("hidden");
        camActiveMode.innerText = "Direct DroidCam Feed";
        streamImg.src = streamInput.value;
        camStatusText.innerText = "RAW PHONE FEED";
      }
    }

    function updateCameraStream() {
      const url = streamInput.value.trim();
      localStorage.setItem("droidcam_stream_url", url);
      if (currentStreamMode === "raw") {
        streamImg.src = "";
        setTimeout(() => { streamImg.src = url; }, 100);
      }
    }

    function onStreamLoaded() {
      fallbackBox.classList.add("hidden");
      camStatusPill.className = "text-[11px] font-mono uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded-full flex items-center gap-1.5";
      camStatusText.innerText = currentStreamMode === "hud" ? "AI BOUNDING BOXES ACTIVE" : "RAW PHONE FEED";
    }

    function onStreamError() {
      fallbackBox.classList.remove("hidden");
      camStatusPill.className = "text-[11px] font-mono uppercase bg-rose-500/20 text-rose-400 border border-rose-500/30 px-2.5 py-0.5 rounded-full flex items-center gap-1.5";
      camStatusText.innerText = currentStreamMode === "hud" ? "AI FEED WAITING" : "OFFLINE";
    }

    function toggleCamFullscreen() {
      const wrapper = document.getElementById("cam-wrapper");
      if (!document.fullscreenElement) {
        wrapper.requestFullscreen().catch(err => console.error(err));
      } else {
        document.exitFullscreen();
      }
    }

    // ========================================================
    // SEQUENTIAL CHECKLIST RENDERER
    // - Detected states: Green Tick Mark
    // - Upcoming state: Loading Spinner
    // ========================================================
    function renderSteps(detectedIdx) {
      const container = document.getElementById("steps-container");
      container.innerHTML = "";

      STEP_NAMES.forEach((name, idx) => {
        let bg = "bg-slate-950/40 border-slate-800 text-slate-500";
        let icon = '<i class="fa-regular fa-circle text-slate-600 text-xs"></i>';
        let badge = '<span class="text-[9px] font-medium uppercase tracking-wider text-slate-600">PENDING</span>';

        if (idx <= detectedIdx) {
          // Detected state: show green tick mark
          bg = "bg-emerald-500/10 border-emerald-500/40 text-emerald-300 ring-1 ring-emerald-500/20 shadow-[0_0_12px_rgba(16,185,129,0.15)]";
          icon = '<i class="fa-solid fa-check text-emerald-400 font-extrabold text-sm"></i>';
          badge = '<span class="text-[9px] font-bold uppercase tracking-wider text-emerald-400 bg-emerald-500/20 border border-emerald-500/30 px-2 py-0.5 rounded-full">DONE</span>';
        } else if (idx === detectedIdx + 1 && detectedIdx < 8) {
          // Upcoming state: show loading spinner
          bg = "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-2 ring-amber-500/30 shadow-[0_0_15px_rgba(245,158,11,0.2)]";
          icon = '<i class="fa-solid fa-spinner fa-spin text-amber-400 text-sm"></i>';
          badge = '<span class="text-[9px] font-bold uppercase tracking-wider text-amber-300 bg-amber-500/20 border border-amber-500/30 px-2 py-0.5 rounded-full pulse-live">UPCOMING</span>';
        }

        const div = document.createElement("div");
        div.className = `border rounded-xl p-3 flex flex-col items-center justify-between text-center transition-all min-h-[96px] ${bg}`;
        div.innerHTML = `
          <div class="mb-1 flex items-center justify-center h-6">${icon}</div>
          <div class="text-[11px] font-medium leading-tight">${name}</div>
          <div class="mt-2 h-4 flex items-center justify-center">${badge}</div>
        `;
        container.appendChild(div);
      });
    }

    renderSteps(0);

    // ========================================================
    // AUTO-RESET & DASHBOARD POLLING
    // ========================================================
    let lastCompletedCycleId = null;
    let autoResetTimer = null;
    let autoResetCountdown = 0;
    const GOAL_TARGET = 50;

    async function triggerResetCycle() {
      if (autoResetTimer) {
        clearInterval(autoResetTimer);
        autoResetTimer = null;
      }
      try {
        const resp = await fetch("/api/assembly/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });
        if (resp.ok) {
          const res = await resp.json();
          renderSteps(0);
          document.getElementById("stat-step-num").innerText = "0";
          document.getElementById("stat-step-bar").style.width = "0%";
          document.getElementById("current-state-title").innerText = STEP_NAMES[0];
          document.getElementById("stat-cycle-status-text").innerText = "Ready for Next Unit";
          if (res.assembly) {
            document.getElementById("header-cycle-num").innerText = `#${res.assembly.cycle_number}`;
          }
          const banner = document.getElementById("diagnostic-banner");
          banner.className = "bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-start gap-4";
          document.getElementById("banner-icon").className = "p-3 bg-slate-800 text-slate-400 rounded-xl text-xl shrink-0";
          document.getElementById("banner-icon").innerHTML = '<i class="fa-solid fa-circle-info"></i>';
          document.getElementById("banner-status").className = "text-sm font-bold uppercase tracking-wider text-slate-300";
          document.getElementById("banner-status").innerText = "Ready for Assembly";
          document.getElementById("banner-text").innerText = `Station ready for Unit #${res.assembly?.cycle_number || ''}. Waiting for Step 1 components...`;
          await pollDashboard();
        }
      } catch (e) {
        console.error("Reset error:", e);
      }
    }

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

          // AI HUD stream active auto-check
          if (currentStreamMode === "hud") {
            if (latestData.hud_active) {
              fallbackBox.classList.add("hidden");
              camStatusPill.className = "text-[11px] font-mono uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded-full flex items-center gap-1.5";
              camStatusText.innerText = "AI BOUNDING BOXES ACTIVE";
            }
          }

          // 1. Update Daily Goal & Cycles Done Today Cards
          const cyclesDone = latestData.cycles_done_today || 0;
          const goalTarget = latestData.goal_target || GOAL_TARGET;

          document.getElementById("stat-goal-val").innerText = goalTarget;
          document.getElementById("stat-cycles-done-today").innerText = cyclesDone;
          document.getElementById("stat-goal-target-display").innerText = goalTarget;

          const goalPct = Math.min(100, Math.round((cyclesDone / goalTarget) * 100));
          document.getElementById("stat-goal-bar").style.width = `${goalPct}%`;
          document.getElementById("stat-goal-pct").innerText = `${goalPct}%`;
          document.getElementById("stat-today-passed").innerText = latestData.today_passes || 0;
          document.getElementById("stat-today-failed").innerText = latestData.today_fails || 0;

          // Header Cycle Info
          if (latestData.active_cycle_number) {
            document.getElementById("header-cycle-num").innerText = `#${latestData.active_cycle_number}`;
          }

          // 2. Handle Cycle Status & Sequential Checklist Auto-Reset
          const isCompletedCycle = sess && (sess.status === "pass" || sess.status === "completed" || (sess.states_completed === 8 && sess.status !== "in_progress"));

          if (isCompletedCycle) {
            // Completed state: show 8/8 done and schedule auto-reset for next unit
            document.getElementById("stat-step-num").innerText = "8";
            document.getElementById("stat-step-bar").style.width = "100%";
            document.getElementById("stat-cycle-status-text").innerText = "Cycle Completed (PASS)";
            document.getElementById("current-state-title").innerText = "8. Complete 9-Part Assembly (PASS)";
            renderSteps(8);

            if (lastCompletedCycleId !== sess.cycle_id) {
              lastCompletedCycleId = sess.cycle_id;
              autoResetCountdown = 4;

              const banner = document.getElementById("diagnostic-banner");
              banner.className = "bg-emerald-950/40 border border-emerald-500/60 rounded-2xl p-5 flex items-start gap-4 ring-2 ring-emerald-500/30";
              document.getElementById("banner-icon").className = "p-3 bg-emerald-500/20 text-emerald-400 rounded-xl text-xl shrink-0";
              document.getElementById("banner-icon").innerHTML = '<i class="fa-solid fa-trophy"></i>';
              document.getElementById("banner-status").className = "text-sm font-bold uppercase tracking-wider text-emerald-400";
              document.getElementById("banner-status").innerText = "CYCLE COMPLETE & VERIFIED (PASS)";
              document.getElementById("banner-text").innerText = `Assembly unit verified successfully! Resetting checklist for next unit in ${autoResetCountdown}s...`;

              if (autoResetTimer) clearInterval(autoResetTimer);
              autoResetTimer = setInterval(async () => {
                autoResetCountdown--;
                if (autoResetCountdown > 0) {
                  document.getElementById("banner-text").innerText = `Assembly unit verified successfully! Resetting checklist for next unit in ${autoResetCountdown}s...`;
                } else {
                  clearInterval(autoResetTimer);
                  autoResetTimer = null;
                  await triggerResetCycle();
                }
              }, 1000);
            }
          } else {
            // In progress cycle or active events
            if (autoResetTimer) {
              clearInterval(autoResetTimer);
              autoResetTimer = null;
            }
            const activeStep = (latestData.current_step !== undefined)
              ? latestData.current_step
              : ((evt && evt.state_index !== undefined) ? evt.state_index : (sess ? (sess.states_completed || 0) : 0));

            document.getElementById("stat-step-num").innerText = activeStep;
            document.getElementById("stat-step-bar").style.width = `${Math.min(100, (activeStep / 8) * 100)}%`;
            document.getElementById("stat-cycle-status-text").innerText = `Step ${activeStep} / 8 In Progress`;
            document.getElementById("current-state-title").innerText = STEP_NAMES[activeStep] || `Step ${activeStep}`;
            renderSteps(activeStep);
          }

          // 3. Update Overall Metrics
          document.getElementById("stat-pass-rate").innerText = `${latestData.pass_rate}%`;
          document.getElementById("stat-total-cycles").innerText = latestData.total_cycles || 0;

          // 4. Update AI Confidence & Alert Banner (only when cycle is active)
          if (evt && (!sess || sess.status === "in_progress")) {
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
              status.innerText = `VERIFIED: ${STEP_NAMES[evt.state_index] || evt.state_name}`;
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

          // 5. Update Verification Table
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