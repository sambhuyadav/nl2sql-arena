"""
NL2SQL Arena — Analysis DSL Parser
Converts the Analysis DSL into executable SQL (SQLite dialect).

DSL Grammar
───────────
QUERY <table>
  [WHERE  <condition> [AND <condition> ...]]
  [JOIN   <table2> ON <col1> = <col2>]
  [AGGREGATE <fn>(<col>[, <col2>]) AS <alias> [BY <group_col>[, <group_col2>]]]
  [SORT   <col> [ASC|DESC]]
  [LIMIT  <n>]
  [COMPARE previous_period]
  [EXPLAIN <free-text reasoning>]

Supported aggregate functions
  sum, avg, count, min, max, count_distinct, avg_hours

Special function
  avg_hours(col1, col2)  →  AVG((julianday(col1) - julianday(col2)) * 24)
  Computes average elapsed time in hours between two datetime columns.

Supported WHERE operators
  =  !=  >  <  >=  <=  BETWEEN  LIKE  IN  IS NULL  IS NOT NULL

Security
  Never uses eval() or exec().  All SQL is produced via string composition
  from whitelisted patterns.  The database.execute_query() layer enforces
  SELECT-only execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Result Type ──────────────────────────────────────────────────────────────


@dataclass
class ParseResult:
    success: bool
    sql: str = ""
    error: str = ""
    parsed_clauses: Dict[str, Any] = field(default_factory=dict)


# ─── Parser ───────────────────────────────────────────────────────────────────


class DSLParser:
    """Hand-written line-oriented parser for the Analysis DSL."""

    AGGREGATE_FUNCTIONS: frozenset[str] = frozenset(
        {"sum", "avg", "count", "min", "max", "count_distinct", "avg_hours"}
    )

    def parse(self, dsl_text: str) -> ParseResult:
        if not dsl_text or not dsl_text.strip():
            return ParseResult(success=False, error="Empty DSL input.")

        lines: List[str] = [
            ln.strip()
            for ln in dsl_text.strip().splitlines()
            if ln.strip()
        ]

        if not lines:
            return ParseResult(success=False, error="DSL contains no non-empty lines.")

        # ── First line must be QUERY <table> ─────────────────────────────────
        m = re.match(r"^QUERY\s+(\w+)\s*$", lines[0], re.IGNORECASE)
        if not m:
            return ParseResult(
                success=False,
                error=(
                    f"DSL must start with 'QUERY <table>'. "
                    f"Got: {lines[0]!r}"
                ),
            )

        table = m.group(1)
        clauses: Dict[str, Any] = {"table": table}

        # ── Parse remaining clauses ───────────────────────────────────────────
        for line in lines[1:]:
            err = self._parse_clause(line, clauses)
            if err:
                return ParseResult(
                    success=False,
                    error=err,
                    parsed_clauses=clauses,
                )

        # ── Build SQL ─────────────────────────────────────────────────────────
        try:
            sql = self._build_sql(clauses)
        except Exception as exc:
            return ParseResult(
                success=False,
                error=f"SQL compilation error: {exc}",
                parsed_clauses=clauses,
            )

        return ParseResult(success=True, sql=sql, parsed_clauses=clauses)

    # ── Clause Parsers ────────────────────────────────────────────────────────

    def _parse_clause(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        """
        Dispatch to the appropriate clause handler.
        Returns an error string on failure, or None on success.
        """
        keyword = line.split()[0].upper() if line.split() else ""

        if keyword == "WHERE":
            return self._handle_where(line, clauses)
        if keyword == "JOIN":
            return self._handle_join(line, clauses)
        if keyword == "AGGREGATE":
            return self._handle_aggregate(line, clauses)
        if keyword == "SORT":
            return self._handle_sort(line, clauses)
        if keyword == "LIMIT":
            return self._handle_limit(line, clauses)
        if keyword == "COMPARE":
            return self._handle_compare(line, clauses)
        if keyword == "EXPLAIN":
            return self._handle_explain(line, clauses)

        return (
            f"Unknown DSL clause keyword {keyword!r}. "
            "Valid: WHERE JOIN AGGREGATE SORT LIMIT COMPARE EXPLAIN"
        )

    def _handle_where(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        cond = line[5:].strip()
        if not cond:
            return "WHERE clause body cannot be empty."
        clauses["where"] = cond
        return None

    def _handle_join(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        m = re.match(r"^JOIN\s+(\w+)\s+ON\s+(.+)$", line, re.IGNORECASE)
        if not m:
            return (
                "Invalid JOIN syntax. "
                "Expected: JOIN <table> ON <condition>. "
                f"Got: {line!r}"
            )
        clauses["join"] = {"table": m.group(1), "on": m.group(2).strip()}
        return None

    def _handle_aggregate(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        body = line[9:].strip()
        if not body:
            return "AGGREGATE clause body cannot be empty."
        result = self._parse_aggregate_body(body)
        if isinstance(result, str):
            return result  # error message
        clauses["aggregate"] = result
        return None

    def _handle_sort(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        m = re.match(r"^SORT\s+(\S+)(?:\s+(ASC|DESC))?\s*$", line, re.IGNORECASE)
        if not m:
            return (
                "Invalid SORT syntax. "
                "Expected: SORT <col> [ASC|DESC]. "
                f"Got: {line!r}"
            )
        clauses["sort"] = {
            "col": m.group(1),
            "direction": (m.group(2) or "ASC").upper(),
        }
        return None

    def _handle_limit(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        m = re.match(r"^LIMIT\s+(\d+)\s*$", line, re.IGNORECASE)
        if not m:
            return (
                "Invalid LIMIT syntax. "
                "Expected: LIMIT <positive integer>. "
                f"Got: {line!r}"
            )
        clauses["limit"] = int(m.group(1))
        return None

    def _handle_compare(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        m = re.match(r"^COMPARE\s+(.+)$", line, re.IGNORECASE)
        if not m:
            return (
                "Invalid COMPARE syntax. "
                "Expected: COMPARE <period>. "
                f"Got: {line!r}"
            )
        clauses["compare"] = m.group(1).strip()
        return None

    def _handle_explain(self, line: str, clauses: Dict[str, Any]) -> Optional[str]:
        body = line[7:].strip()
        clauses["explain"] = body
        return None

    # ── Aggregate Body Parser ─────────────────────────────────────────────────

    def _parse_aggregate_body(self, text: str) -> Any:
        """
        Parse: <fn>(<args>) AS <alias> [BY <group_cols>]
        Returns a dict on success, an error string on failure.
        """
        m = re.match(
            r"^(\w+)\(([^)]+)\)\s+AS\s+(\w+)(?:\s+BY\s+(.+))?\s*$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return (
                "Invalid AGGREGATE syntax. "
                "Expected: <fn>(<col>) AS <alias> [BY <group_col>]. "
                f"Got: {text!r}"
            )

        func     = m.group(1).lower()
        args_raw = m.group(2)
        alias    = m.group(3)
        grp_raw  = m.group(4)

        if func not in self.AGGREGATE_FUNCTIONS:
            return (
                f"Unknown aggregate function {func!r}. "
                f"Supported: {', '.join(sorted(self.AGGREGATE_FUNCTIONS))}"
            )

        args = [a.strip() for a in args_raw.split(",")]

        if func == "avg_hours":
            if len(args) != 2:
                return (
                    f"avg_hours() requires exactly 2 arguments (col1, col2), "
                    f"got {len(args)}: {args}"
                )

        group_by = grp_raw.strip() if grp_raw else None

        return {
            "func": func,
            "args": args,
            "alias": alias,
            "group_by": group_by,
        }

    # ── SQL Builder ───────────────────────────────────────────────────────────

    def _normalize_where(self, condition: str) -> str:
        """Convert DSL WHERE string to valid SQLite WHERE clause.

        The DSL uses double quotes for string literals; SQL uses single quotes.
        Boolean values and NULL keywords are left unchanged.
        """
        # Replace double-quoted string literals with single-quoted equivalents.
        # This regex matches: "..." while not converting things already in SQL.
        result = re.sub(r'"([^"]*)"', r"'\1'", condition)
        return result

    def _build_agg_expr(self, func: str, args: List[str]) -> str:
        if func == "avg_hours":
            c1, c2 = args[0].strip(), args[1].strip()
            return f"AVG((julianday({c1}) - julianday({c2})) * 24)"
        if func == "count_distinct":
            return f"COUNT(DISTINCT {args[0].strip()})"
        return f"{func.upper()}({args[0].strip()})"

    def _build_sql(self, clauses: Dict[str, Any]) -> str:
        table = clauses["table"]

        # ── SELECT ────────────────────────────────────────────────────────────
        if "aggregate" in clauses:
            agg       = clauses["aggregate"]
            agg_expr  = self._build_agg_expr(agg["func"], agg["args"])
            alias     = agg["alias"]
            group_by  = agg.get("group_by")

            if group_by:
                select_clause = f"{group_by}, {agg_expr} AS {alias}"
            else:
                select_clause = f"{agg_expr} AS {alias}"
        else:
            select_clause = "*"

        sql = f"SELECT {select_clause} FROM {table}"

        # ── JOIN ──────────────────────────────────────────────────────────────
        if "join" in clauses:
            j    = clauses["join"]
            sql += f" JOIN {j['table']} ON {j['on']}"

        # ── WHERE ─────────────────────────────────────────────────────────────
        if "where" in clauses:
            sql += f" WHERE {self._normalize_where(clauses['where'])}"

        # ── GROUP BY ──────────────────────────────────────────────────────────
        if "aggregate" in clauses:
            gb = clauses["aggregate"].get("group_by")
            if gb:
                sql += f" GROUP BY {gb}"

        # ── ORDER BY ──────────────────────────────────────────────────────────
        if "sort" in clauses:
            s    = clauses["sort"]
            sql += f" ORDER BY {s['col']} {s['direction']}"

        # ── LIMIT ─────────────────────────────────────────────────────────────
        if "limit" in clauses:
            sql += f" LIMIT {clauses['limit']}"

        # ── COMPARE (informational comment only) ──────────────────────────────
        if "compare" in clauses:
            sql += f"  -- COMPARE: {clauses['compare']}"

        return sql


# ─── Module-level helper ─────────────────────────────────────────────────────

_parser = DSLParser()


def parse_dsl(dsl_text: str) -> ParseResult:
    """Parse a DSL string and return a ParseResult."""
    return _parser.parse(dsl_text)
