"""
NL2SQL Arena — Shaped Reward Function

compute_reward() is the single entry point for reward computation.
It combines DSL quality signals, grader output, bonuses, and penalties
into one ArenaReward, clamped to [0.0, 1.0].

Component weights
─────────────────
+0.05  valid DSL syntax
+0.10  correct table(s) selected
+0.15  correct WHERE conditions
+0.20  correct aggregation function and column
+0.25  result matches ground truth (from grader)
+0.05  EXPLAIN clause present and non-empty  (bonus)
─────────────────────────────────────────────────────
-0.05  per step penalty for each step AFTER step 3
-0.10  re-submitting identical DSL as previous step
-0.20  SQL execution error (syntax / missing column)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import ArenaAction, ArenaReward
from arena_dsl import ParseResult
from graders import grade_result


def compute_reward(
    action: ArenaAction,
    parse_result: ParseResult,
    execution_result: Optional[List[Dict[str, Any]]],
    sql_error: Optional[str],
    task_id: str,
    ground_truth: Dict[str, Any],
    step: int,
    prev_dsl: Optional[str] = None,
) -> ArenaReward:
    """
    Compute a shaped ArenaReward for a single step.

    Parameters
    ----------
    action          : The agent's submitted action (dsl + optional explain).
    parse_result    : Output of DSLParser.parse(action.dsl).
    execution_result: Rows returned by execute_query(), or None on error/no-parse.
    sql_error       : Error string from SQL execution (or None).
    task_id         : One of 'simple-lookup', 'multi-table-join', 'debug-and-fix'.
    ground_truth    : Pre-computed ground truth dict for this task.
    step            : Current step number (1-indexed).
    prev_dsl        : DSL string submitted in the immediately preceding step.

    Returns
    -------
    ArenaReward with .value clamped to [0.0, 1.0].
    """
    breakdown: Dict[str, float] = {}
    messages: List[str] = []

    clauses = parse_result.parsed_clauses

    # ── 1. DSL Syntax ─────────────────────────────────────────────────────────
    if parse_result.success:
        breakdown["syntax"] = 0.05
        messages.append("Valid DSL syntax (+0.05)")
    else:
        breakdown["syntax"] = 0.0
        messages.append(f"DSL parse error: {parse_result.error}")

    # ── 2. SQL Execution Error ────────────────────────────────────────────────
    if parse_result.success and sql_error:
        breakdown["execution_error"] = -0.20
        messages.append(f"SQL execution error (-0.20): {sql_error[:80]}")

    # ── 3. Repeat Penalty ─────────────────────────────────────────────────────
    if prev_dsl is not None and action.dsl.strip() == prev_dsl.strip():
        breakdown["repeat_penalty"] = -0.10
        messages.append("Identical re-submission (-0.10)")

    # ── 4. Table Selection ────────────────────────────────────────────────────
    if parse_result.success:
        expected_tables = _expected_tables(task_id)
        agent_table = clauses.get("table", "").lower()
        join_table  = clauses.get("join", {}).get("table", "").lower()
        used_tables = {t for t in (agent_table, join_table) if t}

        if used_tables & expected_tables:
            breakdown["table"] = 0.10
            messages.append("Correct table(s) selected (+0.10)")

    # ── 5. WHERE Conditions ───────────────────────────────────────────────────
    if parse_result.success:
        if task_id == "multi-table-join":
            # For this task the correct answer has NO WHERE clause.
            # Reward the agent for not applying a spurious filter.
            if "where" not in clauses:
                breakdown["where"] = 0.15
                messages.append("No spurious WHERE filter (+0.15)")
        elif "where" in clauses:
            where_score = _score_where(task_id, clauses.get("where", ""))
            if where_score > 0:
                breakdown["where"] = where_score
                messages.append(f"WHERE conditions (+{where_score:.2f})")

    # ── 6. Aggregation ────────────────────────────────────────────────────────
    if parse_result.success and "aggregate" in clauses:
        agg_score = _score_aggregation(task_id, clauses["aggregate"])
        if agg_score > 0:
            breakdown["aggregation"] = agg_score
            messages.append(f"Aggregation (+{agg_score:.2f})")

    # ── 7. Result Match (from grader) ─────────────────────────────────────────
    if parse_result.success and execution_result is not None and not sql_error:
        grade, grade_breakdown = grade_result(
            task_id=task_id,
            execution_result=execution_result,
            parsed_clauses=clauses,
            ground_truth=ground_truth,
            parse_success=parse_result.success,
        )
        # Use 0.25 as the cap for grader-sourced result match reward
        result_reward = grade * 0.25
        if result_reward > 0:
            breakdown["result_match"] = round(result_reward, 4)
            messages.append(f"Result match (+{result_reward:.2f}, grade={grade:.2f})")
        # Merge grader sub-breakdown for transparency
        for k, v in grade_breakdown.items():
            breakdown[f"grader.{k}"] = v

    # ── 8. EXPLAIN Bonus ──────────────────────────────────────────────────────
    has_explain = bool(action.explain and action.explain.strip())
    if not has_explain and parse_result.success:
        has_explain = bool(clauses.get("explain", "").strip())
    if has_explain:
        breakdown["explain_bonus"] = 0.05
        messages.append("EXPLAIN clause bonus (+0.05)")

    # ── 9. Step Efficiency Penalty ────────────────────────────────────────────
    if step > 3:
        penalty = -0.05 * (step - 3)
        breakdown["step_penalty"] = round(penalty, 4)
        messages.append(f"Step penalty ({penalty:.2f}, step {step})")

    # ── Compute and Clamp ─────────────────────────────────────────────────────
    raw = sum(v for k, v in breakdown.items() if not k.startswith("grader."))
    clamped = round(max(0.0, min(1.0, raw)), 6)

    breakdown["total_raw"] = round(raw, 6)
    breakdown["total"] = clamped

    message = " | ".join(messages) if messages else "No reward components earned."

    return ArenaReward(
        value=clamped,
        breakdown=breakdown,
        message=message,
    )


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _expected_tables(task_id: str) -> frozenset[str]:
    return {
        "simple-lookup":    frozenset({"orders"}),
        "multi-table-join": frozenset({"orders", "customers"}),
        "debug-and-fix":    frozenset({"support_tickets"}),
    }.get(task_id, frozenset())


def _score_where(task_id: str, where_str: str) -> float:
    """Heuristic: does the WHERE clause reference the expected columns/values?"""
    w = where_str.upper()

    if task_id == "simple-lookup":
        score = 0.0
        if "APAC" in w:
            score += 0.08
        if "2023" in w or "BETWEEN" in w:
            score += 0.07
        return min(score, 0.15)

    if task_id == "multi-table-join":
        # WHERE is optional; no deduction for missing it
        return 0.0

    if task_id == "debug-and-fix":
        score = 0.0
        if "PRIORITY" in w and "HIGH" in w:
            score += 0.10
        if "RESOLVED_AT" in w and "NULL" in w:
            score += 0.05
        return min(score, 0.15)

    return 0.0


def _score_aggregation(task_id: str, agg: Dict[str, Any]) -> float:
    """Heuristic: does the AGGREGATE clause match what the task requires?"""
    func = agg.get("func", "")
    args = str(agg.get("args", [])).lower()

    if task_id == "simple-lookup":
        if func == "sum" and "revenue" in args:
            return 0.20
        if func in {"sum", "avg"}:
            return 0.08

    if task_id == "multi-table-join":
        if func == "sum" and "revenue" in args:
            grp = (agg.get("group_by") or "").lower()
            if "name" in grp or "customer" in grp:
                return 0.20
            return 0.12
        if func:
            return 0.05

    if task_id == "debug-and-fix":
        if func == "avg_hours":
            if "resolved_at" in args or "created_at" in args:
                grp = (agg.get("group_by") or "").lower()
                if "issue_type" in grp:
                    return 0.20
                return 0.12
        if func == "avg":
            grp = (agg.get("group_by") or "").lower()
            if "issue_type" in grp:
                return 0.10
        if func:
            return 0.03

    return 0.0
