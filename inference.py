"""
NL2SQL Arena — Baseline Inference Script

Runs all three NL2SQL Arena tasks sequentially using an LLM agent (default:
Qwen/Qwen2.5-72B-Instruct via the HuggingFace router) and emits structured
log lines that the evaluator can parse.

Environment Variables
─────────────────────
API_BASE_URL   LLM router base URL  (default: https://router.huggingface.co/v1)
MODEL_NAME     Model ID             (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN       Bearer token for the HF router
IMAGE_NAME     Docker image name    (used with from_docker_image() if available)
NL2SQL_TASK    Run only this task   (default: all tasks)
NL2SQL_BENCH   Benchmark name tag   (default: nl2sql-arena)
ENV_BASE_URL   Environment API URL  (default: http://localhost:7860)

Log Format (strictly enforced)
──────────────────────────────
[START] task=<name> env=nl2sql-arena model=<model>
[STEP]  step=<n> action=<dsl> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN      = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
    or ""
)
IMAGE_NAME    = os.environ.get("IMAGE_NAME",    "nl2sql-arena")
NL2SQL_TASK   = os.environ.get("NL2SQL_TASK",   "")          # empty = run all
NL2SQL_BENCH  = os.environ.get("NL2SQL_BENCH",  "nl2sql-arena")
ENV_BASE_URL  = os.environ.get("ENV_BASE_URL",  "http://localhost:7860")

ALL_TASKS = ["simple-lookup", "multi-table-join", "product-revenue-breakdown", "debug-and-fix"]

# ─── LLM Client ───────────────────────────────────────────────────────────────

llm = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are NL2SQL Arena agent. You receive a business question and a database \
schema hint. You must write an Analysis DSL program to answer the question.

DSL syntax — each clause on its own line:
  QUERY <table>
    [WHERE <col> <op> <val> [AND <col> <op> <val>]]
    [JOIN <table> ON <col1> = <col2>]
    [AGGREGATE <fn>(<col>) AS <alias> [BY <group_col>]]
    [SORT <col> [ASC|DESC]]
    [LIMIT <n>]
    [EXPLAIN <your reasoning here>]

Available tables: orders, customers, products, support_tickets
Supported functions: sum, avg, count, min, max, avg_hours
Supported operators: =, !=, >, <, >=, <=, BETWEEN, LIKE, IN, IS NULL, IS NOT NULL

Special function:
  avg_hours(col1, col2)  computes AVG hours between two datetime columns.
  Example: avg_hours(resolved_at, created_at)

CRITICAL GROUPING RULE — read carefully:
  There is NO separate GROUP BY clause in this DSL.
  Grouping is specified with BY at the end of the AGGREGATE line.
  CORRECT:   AGGREGATE avg_hours(resolved_at, created_at) AS avg_hours BY issue_type
  WRONG:     AGGREGATE avg_hours(resolved_at, created_at) AS avg_hours
             GROUP BY issue_type       <-- THIS WILL CAUSE AN ERROR
  WRONG:     AGGREGATE avg_hours(resolved_at, created_at) AS avg_hours
             BY issue_type             <-- THIS WILL ALSO CAUSE AN ERROR
  The BY must be on the SAME LINE as AGGREGATE, after the alias.

Rules:
- Reply with ONLY the DSL program. No prose before or after.
- Use the EXPLAIN clause inside the DSL to show your reasoning.
- If you receive an error in the observation, fix the DSL and resubmit.
- If you receive a broken_dsl in the observation, fix its bugs and resubmit.
- String literals in WHERE use double quotes: WHERE region = "APAC"
- Date ranges use BETWEEN: WHERE order_date BETWEEN "2023-01-01" AND "2023-12-31"
- When done=true in the observation, the episode is over — do not resubmit.
"""

# ─── Environment HTTP Client ──────────────────────────────────────────────────


class EnvClient:
    """Thin HTTP wrapper around the NL2SQL Arena REST API."""

    def __init__(self, base_url: str) -> None:
        self.base_url   = base_url.rstrip("/")
        self.session_id = str(uuid.uuid4())
        self._headers   = {"X-Session-Id": self.session_id, "Content-Type": "application/json"}

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/reset",
            params={"task_id": task_id},
            headers=self._headers,
            timeout=30,
        )
        r.raise_for_status()
        # Update session id from response header if present
        sid = r.headers.get("X-Session-Id")
        if sid:
            self.session_id = sid
            self._headers["X-Session-Id"] = sid
        return r.json()

    def step(self, dsl: str, explain: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action": {"dsl": dsl}}
        if explain:
            payload["action"]["explain"] = explain
        r = requests.post(
            f"{self.base_url}/step",
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ─── Log Helpers ──────────────────────────────────────────────────────────────


def _log_start(task_id: str) -> None:
    print(
        f"[START] task={task_id} env={NL2SQL_BENCH} model={MODEL_NAME}",
        flush=True,
    )


def _log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_str  = error if error else "null"
    done_str = "true" if done else "false"
    # Remove newlines from action to keep the log on one line
    action_oneline = " ".join(action.strip().splitlines())
    print(
        f"[STEP] step={step} action={action_oneline} "
        f"reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def _log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Inference ───────────────────────────────────────────────────────────


def _build_user_message(obs: Dict[str, Any]) -> str:
    """Convert an observation dict into a user message for the LLM."""
    parts = [
        f"Question: {obs['question']}",
        "",
        "Schema:",
        obs.get("schema_hint", ""),
    ]

    if obs.get("broken_dsl"):
        parts += [
            "",
            "A broken DSL has been provided. Fix its bugs:",
            obs["broken_dsl"],
        ]

    if obs.get("last_sql_executed"):
        parts += ["", f"Last SQL executed:\n{obs['last_sql_executed']}"]

    if obs.get("last_result_preview"):
        parts += ["", f"Last result (preview):\n{obs['last_result_preview']}"]

    if obs.get("last_error"):
        parts += ["", f"Error from last step:\n{obs['last_error']}"]

    return "\n".join(parts)


def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return the assistant's reply text."""
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


# ─── Single Task Runner ───────────────────────────────────────────────────────


def run_task(task_id: str) -> Tuple[bool, int, float, List[float]]:
    """
    Run one full episode for task_id.

    Returns
    -------
    (success, steps_taken, final_score, per_step_rewards)
    """
    env     = EnvClient(ENV_BASE_URL)
    obs     = env.reset(task_id)
    rewards: List[float] = []
    done    = False
    step    = 0

    _log_start(task_id)

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while not done:
        step += 1

        # Build user turn
        user_msg = _build_user_message(obs)
        conversation.append({"role": "user", "content": user_msg})

        # Cap history: system prompt + last 6 turns to avoid token overflow
        # with different LLMs used in Phase 2 evaluation
        if len(conversation) > 13:  # 1 system + 6 turns * 2 messages
            conversation = [conversation[0]] + conversation[-12:]

        # LLM generates DSL
        try:
            dsl_response = _call_llm(conversation)
        except Exception as exc:
            dsl_response = (
                f'QUERY orders\n'
                f'  EXPLAIN Failed to call LLM: {exc}'
            )

        # Strip any accidental markdown fences
        dsl_clean = dsl_response.strip()
        if dsl_clean.startswith("```"):
            lines = dsl_clean.splitlines()
            dsl_clean = "\n".join(
                ln for ln in lines if not ln.strip().startswith("```")
            ).strip()

        # Add assistant turn to conversation history
        conversation.append({"role": "assistant", "content": dsl_clean})

        # Submit to environment
        try:
            result  = env.step(dsl_clean)
            obs     = result["observation"]
            reward  = float(result["reward"]["value"])
            done    = bool(result["done"])
            error   = obs.get("last_error")
        except Exception as exc:
            reward  = 0.0
            done    = True
            error   = str(exc)
            obs     = {}

        rewards.append(reward)
        _log_step(step, dsl_clean, reward, done, error)

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success     = final_score >= 0.5

    _log_end(success, step, final_score, rewards)
    return success, step, final_score, rewards


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    tasks_to_run = [NL2SQL_TASK] if NL2SQL_TASK else ALL_TASKS

    # Verify environment is reachable
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        if not health.get("db_ready"):
            print("[WARN] Environment reports db_ready=false. Results may be empty.", flush=True)
    except Exception as exc:
        print(f"[ERROR] Cannot reach environment at {ENV_BASE_URL}: {exc}", flush=True)
        return 1

    all_scores: List[float] = []
    exit_code = 0

    for task_id in tasks_to_run:
        try:
            success, steps, score, rewards = run_task(task_id)
            all_scores.append(score)
            if not success:
                exit_code = 1
        except Exception:
            traceback.print_exc(file=sys.stderr)
            _log_end(success=False, steps=0, score=0.0, rewards=[])
            all_scores.append(0.0)
            exit_code = 1

    if len(tasks_to_run) > 1 and all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n[SUMMARY] tasks={len(tasks_to_run)} avg_score={overall:.3f}", flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
