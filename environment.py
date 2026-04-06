"""
NL2SQL Arena — Core Episode Logic

ArenaEnv encapsulates one episode (one session).  Each session instance
holds its own step counter, last SQL, last error, and previous DSL so that
concurrent callers are fully isolated.

Thread-safety: each ArenaEnv carries an asyncio.Lock; callers MUST acquire
it before calling reset() or step().  The server handles locking externally
via session_locks so that the env methods themselves stay synchronous and
easy to unit-test.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from models import ArenaAction, ArenaObservation, ArenaReward
from arena_dsl import DSLParser
from database import execute_query, SCHEMA_HINT
from tasks import get_task, TaskDefinition
from rewards import compute_reward


class ArenaEnv:
    """One independent RL episode for an NL2SQL Arena session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.lock = asyncio.Lock()

        # Episode state — reset on each call to reset()
        self._task: Optional[TaskDefinition] = None
        self._step_count: int = 0
        self._done: bool = False
        self._prev_dsl: Optional[str] = None
        self._last_sql: Optional[str] = None
        self._last_result: Optional[List[Dict[str, Any]]] = None
        self._last_error: Optional[str] = None
        self._current_obs: Optional[ArenaObservation] = None

        self._parser = DSLParser()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "simple-lookup") -> ArenaObservation:
        """Start a new episode for the given task. Returns the initial observation."""
        task = get_task(task_id)  # raises ValueError for unknown task_id

        self._task = task
        self._step_count = 0
        self._done = False
        self._prev_dsl = None
        self._last_sql = None
        self._last_result = None
        self._last_error = None

        obs = self._make_observation()
        self._current_obs = obs
        return obs

    def step(
        self, action: ArenaAction
    ) -> Tuple[ArenaObservation, ArenaReward, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Returns
        -------
        observation : Updated ArenaObservation.
        reward      : Shaped ArenaReward.
        done        : True when the episode has ended.
        info        : Diagnostic dict (parse result, grader notes, etc.).
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        task = self._task

        # ── 1. Parse DSL ──────────────────────────────────────────────────────
        parse_result = self._parser.parse(action.dsl)

        execution_result: Optional[List[Dict[str, Any]]] = None
        sql_error: Optional[str] = None

        # ── 2. Execute SQL (SELECT-only) ──────────────────────────────────────
        if parse_result.success:
            compiled_sql = parse_result.sql
            # Extra security gate: reject non-SELECT before hitting the DB
            sql_upper = compiled_sql.strip().upper()
            if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
                sql_error = (
                    "Security violation: compiled SQL is not a SELECT statement. "
                    "DSL must not produce INSERT/UPDATE/DELETE/DROP."
                )
                self._last_error = sql_error
                self._last_sql = compiled_sql
            else:
                try:
                    execution_result = execute_query(compiled_sql)
                    self._last_sql    = compiled_sql
                    self._last_result = execution_result
                    self._last_error  = None
                except Exception as exc:
                    sql_error        = str(exc)
                    self._last_error = sql_error
                    self._last_sql   = compiled_sql
                    self._last_result = None
        else:
            self._last_error = parse_result.error

        # ── 3. Compute Reward ─────────────────────────────────────────────────
        reward = compute_reward(
            action=action,
            parse_result=parse_result,
            execution_result=execution_result,
            sql_error=sql_error,
            task_id=task.task_id,
            ground_truth=task.ground_truth,
            step=self._step_count,
            prev_dsl=self._prev_dsl,
        )

        self._prev_dsl = action.dsl

        # ── 4. Check Terminal Condition ───────────────────────────────────────
        # End early on a high-quality result so the agent isn't penalised for
        # re-submitting a correct answer while waiting for max_steps.
        # Threshold 0.60: covers Task1 (max 0.80), Task2 (max 0.80), Task3 (max 0.80).
        if self._step_count >= task.max_steps or reward.value >= 0.60:
            self._done = True

        # ── 5. Build Observation ──────────────────────────────────────────────
        obs = self._make_observation()
        self._current_obs = obs

        info: Dict[str, Any] = {
            "parse_success": parse_result.success,
            "parse_error":   parse_result.error,
            "compiled_sql":  parse_result.sql,
            "sql_error":     sql_error,
            "step":          self._step_count,
            "max_steps":     task.max_steps,
        }

        return obs, reward, self._done, info

    # ── State Access ──────────────────────────────────────────────────────────

    @property
    def current_observation(self) -> Optional[ArenaObservation]:
        return self._current_obs

    @property
    def task_id(self) -> Optional[str]:
        return self._task.task_id if self._task else None

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def done(self) -> bool:
        return self._done

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_observation(self) -> ArenaObservation:
        task = self._task
        assert task is not None

        # Schema hint = global schema + task-specific hint
        full_schema = SCHEMA_HINT + task.schema_hint_extra

        # Last result preview: first 3 rows formatted as a compact string
        preview: Optional[str] = None
        if self._last_result:
            rows = self._last_result[:3]
            lines = [str(r) for r in rows]
            if len(self._last_result) > 3:
                lines.append(f"... ({len(self._last_result)} rows total)")
            preview = "\n".join(lines)

        return ArenaObservation(
            task_id=task.task_id,
            question=task.question,
            schema_hint=full_schema,
            broken_dsl=task.broken_dsl,             # None for tasks 1 & 2
            last_sql_executed=self._last_sql,
            last_result_preview=preview,
            last_error=self._last_error,
            step_count=self._step_count,
            done=self._done,
        )
