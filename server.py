"""
NL2SQL Arena — FastAPI Server

Endpoints
─────────
POST /reset         Start a new episode (optional ?task_id=)
POST /step          Submit an action, receive observation + reward
GET  /state         Inspect current session state
GET  /health        Health check (always 200)
GET  /metrics       Aggregate statistics across all episodes
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from models import (
    ArenaObservation,
    ArenaReward,
    HealthResponse,
    MetricsResponse,
    ResetRequest,
    StateResponse,
    StepRequest,
    StepResponse,
)
from environment import ArenaEnv
from database import is_db_ready
from tasks import list_tasks

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NL2SQL Arena",
    description=(
        "An interactive RL arena for training agents to reason over structured data "
        "— from natural language to SQL, through a structured Analysis DSL."
    ),
    version="1.0.0",
)

# ─── Session Registry ────────────────────────────────────────────────────────

MAX_CONCURRENT_ENVS = 32

_sessions: Dict[str, ArenaEnv]         = {}
_session_locks: Dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()          # guards _sessions / _session_locks dicts

# ─── Global Metrics ───────────────────────────────────────────────────────────

_metrics_lock      = asyncio.Lock()
_total_episodes    = 0
_completed_episodes = 0                  # episodes that reached done=True
_task_scores: Dict[str, list[float]]   = defaultdict(list)


async def _record_episode_start(task_id: str) -> None:
    global _total_episodes
    async with _metrics_lock:
        _total_episodes += 1


async def _record_episode_end(task_id: str, final_score: float, done: bool) -> None:
    global _completed_episodes
    async with _metrics_lock:
        _task_scores[task_id].append(final_score)
        if done:
            _completed_episodes += 1


# ─── Session Helpers ─────────────────────────────────────────────────────────


async def _get_or_create_session(session_id: str) -> ArenaEnv:
    async with _registry_lock:
        if session_id not in _sessions:
            # Evict oldest session if at capacity
            if len(_sessions) >= MAX_CONCURRENT_ENVS:
                oldest_id = next(iter(_sessions))
                del _sessions[oldest_id]
                del _session_locks[oldest_id]
            _sessions[session_id]      = ArenaEnv(session_id)
            _session_locks[session_id] = asyncio.Lock()
        return _sessions[session_id]


def _resolve_session_id(x_session_id: Optional[str]) -> str:
    """Return provided session-id or generate a fresh UUID."""
    return (x_session_id or "").strip() or str(uuid.uuid4())


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=ArenaObservation, summary="Start a new episode")
async def reset(
    task_id: Optional[str] = Query(default="simple-lookup"),
    x_session_id: Optional[str] = Header(default=None),
) -> ArenaObservation:
    """
    Reset the environment and return the initial observation.

    Query parameters
    ----------------
    task_id : one of 'simple-lookup' | 'multi-table-join' | 'debug-and-fix'
              Defaults to 'simple-lookup'.

    Headers
    -------
    X-Session-Id : optional session identifier (UUID generated if omitted).
    """
    resolved_task = task_id or "simple-lookup"
    sid           = _resolve_session_id(x_session_id)
    env           = await _get_or_create_session(sid)

    valid_ids = {t.task_id for t in list_tasks()}
    if resolved_task not in valid_ids:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id {resolved_task!r}. Valid: {sorted(valid_ids)}",
        )

    async with _session_locks[sid]:
        obs = env.reset(resolved_task)

    await _record_episode_start(resolved_task)

    # Embed session_id in response headers for client convenience
    content = obs.model_dump()
    response = JSONResponse(content=content)
    response.headers["X-Session-Id"] = sid
    return response  # type: ignore[return-value]


@app.post("/step", response_model=StepResponse, summary="Submit one DSL action")
async def step(
    body: StepRequest,
    x_session_id: Optional[str] = Header(default=None),
) -> StepResponse:
    """
    Submit an ArenaAction and receive the next observation, reward, done flag,
    and diagnostic info.

    Headers
    -------
    X-Session-Id : must match the id returned by /reset.
    """
    sid = _resolve_session_id(x_session_id)

    if sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session {sid!r} not found. "
                "Call POST /reset first to create a session."
            ),
        )

    env = _sessions[sid]

    async with _session_locks[sid]:
        try:
            obs, reward, done, info = env.step(body.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    if done:
        await _record_episode_end(
            task_id=env.task_id or "unknown",
            final_score=reward.value,
            done=done,
        )

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse, summary="Inspect current session state")
async def state(
    x_session_id: Optional[str] = Header(default=None),
) -> StateResponse:
    sid = _resolve_session_id(x_session_id)

    if sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {sid!r} not found.",
        )

    env = _sessions[sid]
    return StateResponse(
        session_id=sid,
        task_id=env.task_id,
        step_count=env.step_count,
        done=env.done,
        current_observation=env.current_observation,
    )


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Always returns 200 OK.  db_ready indicates whether the DB has been seeded."""
    return HealthResponse(
        status="ok",
        env="nl2sql-arena",
        db_ready=is_db_ready(),
    )


@app.get("/metrics", response_model=MetricsResponse, summary="Aggregate statistics")
async def metrics() -> MetricsResponse:
    async with _metrics_lock:
        avg_score: Dict[str, float] = {}
        for task_id, scores in _task_scores.items():
            avg_score[task_id] = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Ensure all tasks appear in the output
        for t in list_tasks():
            avg_score.setdefault(t.task_id, 0.0)

        completion_rate = (
            round(_completed_episodes / _total_episodes, 4)
            if _total_episodes > 0
            else 0.0
        )

        return MetricsResponse(
            total_episodes=_total_episodes,
            avg_score=avg_score,
            completion_rate=completion_rate,
            env="nl2sql-arena",
            version="1.0.0",
        )


# ─── Root redirect ────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "env": "nl2sql-arena",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def main() -> None:
    """Entry point for the [project.scripts] console script."""
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, workers=1)
