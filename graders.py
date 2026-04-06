"""
NL2SQL Arena — Per-Task Grader Functions

Each grader inspects the execution result against the pre-computed ground
truth and returns a RESULT QUALITY SCORE in [0.0, 1.0].

This score is used by rewards.py as the "result_match" component:
    result_match_reward = grade * 0.25

Structural quality (table selection, WHERE conditions, aggregation choice)
is handled separately in rewards.py — graders focus only on whether the
output data is correct.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _first_numeric(row: Dict[str, Any]) -> Optional[float]:
    """Return the first numeric value found in a result row."""
    for v in row.values():
        if isinstance(v, (int, float)) and v is not None:
            return float(v)
    return None


def _name_from_row(row: Dict[str, Any]) -> str:
    """Extract a normalised customer name string from a result row."""
    for key in ("name", "customer_name", "Name", "NAME"):
        if key in row and isinstance(row[key], str):
            return row[key].lower().strip()
    # Fall back to first string value longer than 2 chars
    for v in row.values():
        if isinstance(v, str) and len(v) > 2:
            return v.lower().strip()
    return ""


# ─── Task 1: simple-lookup ────────────────────────────────────────────────────


def grade_simple_lookup(
    execution_result: Optional[List[Dict[str, Any]]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Returns 1.0  if the aggregated value matches GT within 1 % tolerance.
    Returns 0.5  if the value is in the right ballpark (within 30 %).
    Returns 0.0  if no result, null value, or completely wrong.
    """
    if not execution_result:
        return 0.0

    row = execution_result[0]
    value = _first_numeric(row)
    gt_value = ground_truth.get("value")

    if value is None or gt_value is None:
        return 0.0

    gt_f = float(gt_value)
    if gt_f == 0.0:
        return 1.0 if value == 0.0 else 0.0

    relative_err = abs(value - gt_f) / abs(gt_f)
    if relative_err <= 0.01:   # within 1 %
        return 1.0
    if relative_err <= 0.30:   # within 30 % — right column, wrong filter
        return 0.5
    return 0.0


# ─── Task 2: multi-table-join ─────────────────────────────────────────────────


def grade_multi_table_join(
    execution_result: Optional[List[Dict[str, Any]]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Positional match: fraction of the top-5 rows that match GT by customer name
    in the correct position.  Returns 0.0–1.0.
    """
    gt_rows: List[Dict[str, Any]] = ground_truth.get("top5", [])
    if not gt_rows or not execution_result:
        return 0.0

    gt_names     = [_name_from_row(r) for r in gt_rows]
    result_names = [_name_from_row(r) for r in execution_result[:5]]

    if not gt_names:
        return 0.0

    # Positional match (full credit)
    positional = sum(
        1 for g, r in zip(gt_names, result_names) if g == r and g != ""
    )

    # Unordered match (partial credit: 60 % of unordered matches)
    unordered = sum(1 for n in result_names if n in gt_names and n != "")

    if positional == len(gt_names):
        return 1.0
    if positional > 0:
        return positional / len(gt_names)
    # Any correct names present, just wrong order?
    if unordered > 0:
        return 0.6 * unordered / len(gt_names)
    # Result is present but names don't match at all
    if execution_result:
        return 0.05
    return 0.0


# ─── Task 3: debug-and-fix ────────────────────────────────────────────────────


def grade_debug_and_fix(
    execution_result: Optional[List[Dict[str, Any]]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Fraction of issue_type groups whose avg_resolution_hours matches GT
    within a 2-hour tolerance.  Returns 0.0–1.0.
    """
    gt_by_type: Dict[str, float] = ground_truth.get("by_type", {})
    if not gt_by_type or not execution_result:
        # At least reward returning any rows for this task
        return 0.05 if execution_result else 0.0

    # Build {issue_type: avg_hours} from execution result
    result_by_type: Dict[str, float] = {}
    for row in execution_result:
        issue = row.get("issue_type")
        avg_val: Optional[float] = None
        for key, val in row.items():
            if key != "issue_type" and isinstance(val, (int, float)):
                avg_val = float(val)
        if issue and avg_val is not None:
            result_by_type[issue] = avg_val

    if not result_by_type:
        return 0.05  # returned rows but couldn't parse them

    total = len(gt_by_type)
    matched = sum(
        1
        for issue, gt_val in gt_by_type.items()
        if issue in result_by_type and abs(result_by_type[issue] - gt_val) <= 2.0
    )
    return matched / total


# ─── Router ───────────────────────────────────────────────────────────────────


def grade_result(
    task_id: str,
    execution_result: Optional[List[Dict[str, Any]]],
    parsed_clauses: Dict[str, Any],
    ground_truth: Dict[str, Any],
    parse_success: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Route to the correct grader.

    Returns
    -------
    (grade, breakdown)
        grade     : result quality score in [0.0, 1.0]
        breakdown : single-entry dict for transparency in reward breakdown
    """
    if task_id == "simple-lookup":
        g = grade_simple_lookup(execution_result, ground_truth)
    elif task_id == "multi-table-join":
        g = grade_multi_table_join(execution_result, ground_truth)
    elif task_id == "debug-and-fix":
        g = grade_debug_and_fix(execution_result, ground_truth)
    else:
        g = 0.0

    return g, {"result_quality": round(g, 4)}
