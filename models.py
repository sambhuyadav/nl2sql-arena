"""
NL2SQL Arena — Pydantic v2 Data Models
All request/response shapes for the OpenEnv environment.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Core RL Models ──────────────────────────────────────────────────────────


class ArenaObservation(BaseModel):
    task_id: str
    question: str                        # Natural language business question
    schema_hint: str                     # Table schema summary as string
    broken_dsl: Optional[str] = None     # Only for Task 3 — the buggy DSL
    last_sql_executed: Optional[str] = None   # SQL from last step (feedback)
    last_result_preview: Optional[str] = None # First 3 rows of last result as string
    last_error: Optional[str] = None     # SQL execution or DSL parse error
    step_count: int
    done: bool


class ArenaAction(BaseModel):
    dsl: str                             # The agent's DSL program
    explain: Optional[str] = None        # Optional explanation (earns bonus)


class ArenaReward(BaseModel):
    value: float                         # 0.0–1.0 clamped
    breakdown: dict[str, Any]            # e.g. {"syntax": 0.05, "table": 0.10}
    message: str                         # Human-readable feedback


# ─── API Request/Response Models ─────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: Optional[str] = "simple-lookup"


class StepRequest(BaseModel):
    action: ArenaAction


class StepResponse(BaseModel):
    observation: ArenaObservation
    reward: ArenaReward
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    task_id: Optional[str] = None
    step_count: int
    done: bool
    current_observation: Optional[ArenaObservation] = None


class HealthResponse(BaseModel):
    status: str
    env: str
    db_ready: bool


class MetricsResponse(BaseModel):
    total_episodes: int
    avg_score: dict[str, float]
    completion_rate: float
    env: str
    version: str
