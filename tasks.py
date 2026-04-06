"""
NL2SQL Arena — Task Definitions and Ground Truth

Ground truths are computed once at import time from the live database,
ensuring determinism across all evaluation runs on the same seeded DB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Task Definition ──────────────────────────────────────────────────────────


@dataclass
class TaskDefinition:
    task_id: str
    difficulty: str
    max_steps: int
    question: str
    schema_hint_extra: str          # task-specific hints appended to global schema
    broken_dsl: Optional[str]       # only Task 4
    ground_truth_sql: str           # authoritative SQL for computing ground truth
    ground_truth: Dict[str, Any] = field(default_factory=dict)  # populated at runtime


# ─── Task Registry ────────────────────────────────────────────────────────────


TASKS: Dict[str, TaskDefinition] = {
    "simple-lookup": TaskDefinition(
        task_id="simple-lookup",
        difficulty="easy",
        max_steps=5,
        question=(
            "What is the total revenue for the APAC region in 2023?"
        ),
        schema_hint_extra=(
            "\nFocus table: orders\n"
            "Hint: filter on region = 'APAC' and order_date in 2023."
        ),
        broken_dsl=None,
        ground_truth_sql=(
            "SELECT SUM(revenue) AS total_revenue "
            "FROM orders "
            "WHERE region = 'APAC' "
            "  AND order_date BETWEEN '2023-01-01' AND '2023-12-31'"
        ),
    ),

    "multi-table-join": TaskDefinition(
        task_id="multi-table-join",
        difficulty="medium",
        max_steps=8,
        question=(
            "List the top 5 customers by total revenue, "
            "showing their name, country, and total spend."
        ),
        schema_hint_extra=(
            "\nFocus tables: orders JOIN customers\n"
            "Hint: join on customer_id, aggregate sum(revenue), "
            "group by customer name and country, sort DESC, limit 5."
        ),
        broken_dsl=None,
        ground_truth_sql=(
            "SELECT customers.name, customers.country, "
            "       SUM(orders.revenue) AS total_revenue "
            "FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id "
            "GROUP BY customers.customer_id, customers.name, customers.country "
            "ORDER BY total_revenue DESC "
            "LIMIT 5"
        ),
    ),

    "product-revenue-breakdown": TaskDefinition(
        task_id="product-revenue-breakdown",
        difficulty="medium",
        max_steps=8,
        question=(
            "Which product categories generated the highest average revenue per order "
            "in 2023? Rank all categories from highest to lowest."
        ),
        schema_hint_extra=(
            "\nFocus tables: orders JOIN products\n"
            "Hint: join on product_id, filter order_date in 2023, "
            "aggregate avg(revenue), group by products.category, sort DESC."
        ),
        broken_dsl=None,
        ground_truth_sql=(
            "SELECT products.category, AVG(orders.revenue) AS avg_revenue "
            "FROM orders "
            "JOIN products ON orders.product_id = products.product_id "
            "WHERE orders.order_date BETWEEN '2023-01-01' AND '2023-12-31' "
            "GROUP BY products.category "
            "ORDER BY avg_revenue DESC"
        ),
    ),

    "debug-and-fix": TaskDefinition(
        task_id="debug-and-fix",
        difficulty="hard",
        max_steps=10,
        question=(
            "Find the average resolution time in hours for high-priority "
            "support tickets, grouped by issue type."
        ),
        schema_hint_extra=(
            "\nFocus table: support_tickets\n"
            "Hint: filter priority = 'high', compute time diff between "
            "resolved_at and created_at in hours using avg_hours(), "
            "group by issue_type.\n"
            "Available DSL function: avg_hours(col1, col2) — average hours between two datetime columns."
        ),
        # Bug 1: wrong column name `resolution_time` (does not exist)
        # Bug 2: missing BY issue_type in AGGREGATE
        broken_dsl=(
            'QUERY support_tickets\n'
            '  WHERE priority = "high"\n'
            '  AGGREGATE avg(resolution_time) AS avg_resolution\n'
            '  SORT avg_resolution DESC'
        ),
        ground_truth_sql=(
            "SELECT issue_type, "
            "AVG((julianday(resolved_at) - julianday(created_at)) * 24) "
            "    AS avg_resolution_hours "
            "FROM support_tickets "
            "WHERE priority = 'high' AND resolved_at IS NOT NULL "
            "GROUP BY issue_type "
            "ORDER BY issue_type"
        ),
    ),
}


# ─── Ground Truth Loader ──────────────────────────────────────────────────────


def _load_ground_truths() -> None:
    """Execute ground-truth SQL for each task and cache results."""
    from database import execute_query, is_db_ready  # local import to avoid circularity

    if not is_db_ready():
        return

    for task in TASKS.values():
        try:
            rows = execute_query(task.ground_truth_sql)
            if task.task_id == "simple-lookup":
                value = rows[0]["total_revenue"] if rows else 0.0
                task.ground_truth = {"value": float(value) if value is not None else 0.0}

            elif task.task_id == "multi-table-join":
                task.ground_truth = {"top5": rows}

            elif task.task_id == "product-revenue-breakdown":
                task.ground_truth = {"by_category": rows}

            elif task.task_id == "debug-and-fix":
                by_type: Dict[str, float] = {}
                for row in rows:
                    issue = row.get("issue_type", "")
                    avg_h = row.get("avg_resolution_hours")
                    if issue and avg_h is not None:
                        by_type[issue] = float(avg_h)
                task.ground_truth = {"by_type": by_type}

        except Exception as exc:
            print(f"[tasks] Warning: could not load ground truth for {task.task_id!r}: {exc}")


_load_ground_truths()


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task {task_id!r}. Valid tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> List[TaskDefinition]:
    return list(TASKS.values())
