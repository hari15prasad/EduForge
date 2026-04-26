"""
app.py — FastAPI wrapper exposing EduForgeEnv for OpenEnv integration.
Updated to utilize Hugging Face Inference Endpoints for the 'Voice' layer.
"""

from __future__ import annotations
import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal imports
from src.environment.openenv_wrapper import EduForgeEnv, Observation, StepInfo
from src.environment.student_fsm import MisconceptionType
from eduforge_runtime import EduForgeRuntime  # Our new optimized bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global State & Lazy Initialisation
# ---------------------------------------------------------------------------
_tutor_runtime = None
_sessions: dict[str, EduForgeEnv] = {}
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", 256))

def _get_runtime() -> EduForgeRuntime:
    global _tutor_runtime
    if _tutor_runtime is None:
        try:
            # Pulling config from Environment Variables for HF Spaces security
            _tutor_runtime = EduForgeRuntime(
                rl_checkpoint=os.getenv("MODEL_PATH", "latest_model.pt"),
                hf_endpoint=os.getenv("HF_ENDPOINT_URL"), # Your $30 credit endpoint
                hf_token=os.getenv("HF_TOKEN"),            # Your HF Write Token
                groq_api_key=os.getenv("GROQ_API_KEY")    # Added for speed/reliability
            )
        except Exception as e:
            logger.error("Could not initialise EduForgeRuntime: %s", e)
            raise HTTPException(status_code=503, detail=f"Tutor runtime unavailable: {e}")
    return _tutor_runtime

def _get_env(session_id: str) -> EduForgeEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed:       Optional[int] = None
    session_id: Optional[str] = None   # client-supplied or auto-generated

class ResetResponse(BaseModel):
    session_id:      str
    student_response: str
    confusion:       float
    attention:       float
    turn:            int
    misconception_id: str

class StepRequest(BaseModel):
    session_id: str
    action:     str    # must contain <STRATEGY>...</STRATEGY>

class StepResponse(BaseModel):
    session_id:       str
    student_response: str
    confusion:        float
    attention:        float
    turn:             int
    misconception_id: str
    reward:           float
    done:             bool
    done_reason:      Optional[str]
    parsed_action:    Optional[str]

class SessionListResponse(BaseModel):
    active_sessions: int
    session_ids:     list[str]

class HybridChatRequest(BaseModel):
    student_message: str
    state_vector: list[float] = [5.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    session_id: str = "default"
    forced_strategy: Optional[str] = None

class HybridChatResponse(BaseModel):
    response: str   # Renamed from 'reply' to match what the frontend expects!
    strategy: str
    q_values: dict = {}
    handling_score: int = 85
    status: str = "success"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _obs_to_reset_response(session_id: str, obs: Observation) -> ResetResponse:
    return ResetResponse(
        session_id=session_id,
        student_response=obs.student_response,
        confusion=obs.confusion,
        attention=obs.attention,
        turn=obs.turn,
        misconception_id=obs.misconception_id.value,
    )

def _build_step_response(
    session_id: str,
    obs: Observation,
    reward: float,
    done: bool,
    info: StepInfo,
) -> StepResponse:
    return StepResponse(
        session_id=session_id,
        student_response=obs.student_response,
        confusion=obs.confusion,
        attention=obs.attention,
        turn=obs.turn,
        misconception_id=obs.misconception_id.value,
        reward=reward,
        done=done,
        done_reason=info.done_reason,
        parsed_action=info.parsed_action.value if info.parsed_action else None,
    )

# ---------------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _sessions.clear()

app = FastAPI(title="EduForge API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/hybrid/chat", response_model=HybridChatResponse, tags=["hybrid"])
async def hybrid_chat(req: HybridChatRequest) -> HybridChatResponse:
    """
    The main entry point for the Next.js Frontend.
    Triggers the RL Brain -> HF Voice pipeline.
    """
    runtime = _get_runtime()
    
    # Run the hybrid pipeline
    reply, strategy, q_values_list = runtime.generate_tutor_response(
        user_message=req.student_message,
        state_vector=req.state_vector,
        forced_strategy=req.forced_strategy,
        session_id=req.session_id
    )
    
    # Calculate handling score dynamically based on expected reward (max Q-value)
    # Assumes Q-values range roughly from -20 to 100
    max_q = max(q_values_list) if q_values_list else 0
    calculated_score = max(0, min(100, int((max_q + 20) / 1.2)))
    
    q_values_dict = {str(i): v for i, v in enumerate(q_values_list)}
    
    # We map 'reply' to 'response' so it doesn't break the Next.js frontend code
    return HybridChatResponse(
        response=reply,
        strategy=strategy,
        q_values=q_values_dict,
        handling_score=calculated_score
    )

@app.get("/health")
def health():
    return {"status": "active", "sessions": len(_sessions), "runtime_loaded": _tutor_runtime is not None}

@app.get("/sessions", response_model=SessionListResponse, tags=["meta"])
def list_sessions() -> SessionListResponse:
    return SessionListResponse(
        active_sessions=len(_sessions),
        session_ids=list(_sessions.keys()),
    )

@app.post("/reset", response_model=ResetResponse, tags=["env"])
def reset(req: ResetRequest) -> ResetResponse:
    if len(_sessions) >= MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]

    session_id = req.session_id or str(uuid.uuid4())
    env = EduForgeEnv(seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env

    return _obs_to_reset_response(session_id, obs)

@app.post("/step", response_model=StepResponse, tags=["env"])
def step(req: StepRequest) -> StepResponse:
    env = _get_env(req.session_id)
    obs, reward, done, info = env.step(req.action)
    return _build_step_response(req.session_id, obs, reward, done, info)

@app.delete("/sessions/{session_id}", tags=["meta"])
def delete_session(session_id: str) -> dict[str, str]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"deleted": session_id}

# ---------------------------------------------------------------------------
# Hugging Face Spaces Entrypoint (Frontend & Backend)
# ---------------------------------------------------------------------------
# We mount the static files LAST so they don't mask the API endpoints above
if os.path.exists("frontend_static"):
    app.mount("/", StaticFiles(directory="frontend_static", html=True), name="static")

# ---------------------------------------------------------------------------
# Hugging Face Spaces entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )