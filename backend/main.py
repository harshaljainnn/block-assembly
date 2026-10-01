import os
import secrets
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
CV_API_KEY = os.getenv("CV_API_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is missing")

if not CV_API_KEY:
    raise RuntimeError("CV_API_KEY is missing")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Assembly Performance API",
    description="Backend API for the Assembly Performance Dashboard",
    version="1.0.0",
)


# Allow the React dashboard to call the API later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SUPABASE HELPERS
# ============================================================

SUPABASE_REST_URL = f"{SUPABASE_URL.rstrip('/')}/rest/v1"


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
    Small helper for communicating with Supabase REST API.
    """

    url = f"{SUPABASE_REST_URL}/{table}"

    response = requests.request(
        method=method,
        url=url,
        headers=supabase_headers(),
        params=params,
        json=json_data,
        timeout=10,
    )

    if not response.ok:
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
    Only the CV system should be able to send assembly data.
    """

    if not api_key or not secrets.compare_digest(api_key, CV_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid CV API key",
        )


# ============================================================
# MODELS
# ============================================================

class StartAssemblyRequest(BaseModel):
    operator_id: str = Field(..., description="Operator code, e.g. OP002")


class EventRequest(BaseModel):
    cycle_id: str

    state_index: Optional[int] = None
    state_name: Optional[str] = None
    state_title: Optional[str] = None

    status: str = Field(
        ...,
        description="holding, advanced, error or completed"
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
        description="pass or fail"
    )

    failure_reason: Optional[str] = None

    states_completed: Optional[int] = None
    total_states: int = 9


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "assembly-performance-api",
        "supabase_configured": bool(SUPABASE_URL),
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
    # Find operator using operator_code
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

    # --------------------------------------------------------
    # Generate cycle ID
    # --------------------------------------------------------

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(2).upper()

    cycle_id = f"ASM-{timestamp}-{random_part}"

    # --------------------------------------------------------
    # Create assembly session
    # --------------------------------------------------------

    assembly = supabase_request(
        "POST",
        "assembly_sessions",
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
    # Find assembly session
    # --------------------------------------------------------

    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": "id,cycle_id,states_completed,total_states,status",
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    # --------------------------------------------------------
    # Insert event
    # --------------------------------------------------------

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
        json_data=event_data,
    )

    # --------------------------------------------------------
    # Update completed state count
    # --------------------------------------------------------

    current_completed = assembly.get("states_completed") or 0

    new_completed = current_completed

    if (
        data.status in ["advanced", "completed"]
        and data.state_index is not None
    ):
        new_completed = max(
            current_completed,
            data.state_index
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
    # Find assembly
    # --------------------------------------------------------

    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": "id,cycle_id,start_time,states_completed,total_states",
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    # --------------------------------------------------------
    # Calculate duration
    # --------------------------------------------------------

    start_time = datetime.fromisoformat(
        assembly["start_time"].replace("Z", "+00:00")
    )

    end_time = datetime.now(timezone.utc)

    duration_seconds = max(
        0,
        int((end_time - start_time).total_seconds())
    )

    # --------------------------------------------------------
    # Final state count
    # --------------------------------------------------------

    states_completed = (
        data.states_completed
        if data.states_completed is not None
        else assembly.get("states_completed", 0)
    )

    # --------------------------------------------------------
    # Validate status
    # --------------------------------------------------------

    status = data.status.lower()

    if status not in ["pass", "fail"]:
        raise HTTPException(
            status_code=400,
            detail="status must be 'pass' or 'fail'",
        )

    # --------------------------------------------------------
    # Update assembly
    # --------------------------------------------------------

    updated = supabase_request(
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
