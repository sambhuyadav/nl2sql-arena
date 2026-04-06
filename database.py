"""
NL2SQL Arena — Database Layer
SQLite setup, synthetic seed data, query executor.

Run directly to (re)create and seed the database:
    python database.py
"""

from __future__ import annotations

import os
import re
import random
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl2sql_arena.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    email       TEXT,
    country     TEXT,
    segment     TEXT,
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT   NOT NULL,
    category   TEXT,
    unit_price REAL,
    cost       REAL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id  INTEGER,
    quantity    INTEGER,
    revenue     REAL,
    order_date  TEXT,
    region      TEXT,
    status      TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id   INTEGER PRIMARY KEY,
    customer_id INTEGER,
    issue_type  TEXT,
    priority    TEXT,
    status      TEXT,
    created_at  TEXT,
    resolved_at TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SCHEMA_HINT = """\
Tables:
- orders(order_id, customer_id, product_id, quantity, revenue, order_date, region, status)
- customers(customer_id, name, email, country, segment, signup_date)
- products(product_id, name, category, unit_price, cost)
- support_tickets(ticket_id, customer_id, issue_type, priority, status, created_at, resolved_at)

Enumerations:
- orders.region    : APAC, EMEA, AMER, LATAM
- orders.status    : completed, pending, cancelled, refunded
- customers.segment: Enterprise, SMB, Startup, Government, Education
- customers.country: USA, UK, Germany, France, Japan, Australia, Canada, India, Brazil, Singapore
- support_tickets.priority  : low, medium, high, critical
- support_tickets.issue_type: billing, technical, account, feature-request, security
- support_tickets.status    : open, resolved, closed, escalated

Date ranges: 2022-01-01 to 2024-12-31
Foreign keys:
  orders.customer_id -> customers.customer_id
  orders.product_id  -> products.product_id
  support_tickets.customer_id -> customers.customer_id\
"""

# ─── Connection ───────────────────────────────────────────────────────────────


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─── Query Executor ───────────────────────────────────────────────────────────

# Forbidden DML/DDL keywords (whole-word match)
_DANGEROUS_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def execute_query(sql: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Execute a SELECT-only query and return results as a list of dicts.

    Raises ValueError for any non-SELECT or dangerous SQL.
    Raises sqlite3.Error on execution failure.
    """
    sql_stripped = sql.strip()
    sql_upper = sql_stripped.upper().lstrip()

    # Must start with SELECT or WITH (CTE)
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise ValueError(
            "Security violation: only SELECT statements are allowed. "
            f"Got: {sql_stripped[:80]!r}"
        )

    # No dangerous keywords anywhere in the query
    match = _DANGEROUS_PATTERNS.search(sql_stripped)
    if match:
        raise ValueError(
            f"Security violation: forbidden keyword '{match.group()}' detected. "
            "Query rejected."
        )

    conn = get_connection()
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        cursor = conn.cursor()
        cursor.execute(sql_stripped)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def is_db_ready() -> bool:
    return os.path.exists(DB_PATH)


# ─── Seed Data ────────────────────────────────────────────────────────────────

_PRODUCT_NAMES = [
    "DataPilot Pro", "CloudSentinel", "AnalyticsEdge", "SecureVault", "InfraBoost",
    "QueryMaster", "StreamFlow", "LogAnalyzer", "MetricHub", "AlertManager",
    "DataBridge", "CloudOptimizer", "SecurityScanner", "ReportBuilder", "DashboardPro",
    "APIGateway Pro", "CacheEngine", "BackupSolution", "MonitoringKit", "DeployBot",
    "DataWarehouse", "MLPipeline", "ETLEngine", "DataCatalog", "StreamProcessor",
    "NetworkMonitor", "IdentityManager", "ComplianceKit", "CostOptimizer", "SpendAnalyzer",
    "SalesTracker", "CustomerInsights", "MarketingHub", "RevenueOptimizer", "ChurnPredictor",
    "SupportDesk", "TicketManager", "KnowledgeBase", "ChatBot Pro", "EmailAutomation",
    "ProjectPlanner", "TaskTracker", "CollabSuite", "MeetingManager", "DocumentEditor",
    "FileSync Pro", "BackupPro", "DisasterRecovery", "HighAvailability", "LoadBalancer",
]

_COUNTRIES = [
    "USA", "UK", "Germany", "France", "Japan",
    "Australia", "Canada", "India", "Brazil", "Singapore",
]
_SEGMENTS   = ["Enterprise", "SMB", "Startup", "Government", "Education"]
_REGIONS    = ["APAC", "EMEA", "AMER", "LATAM"]
_STATUSES   = ["completed", "pending", "cancelled", "refunded"]
_CATEGORIES = ["Software", "Hardware", "Services", "Cloud", "Security", "Analytics"]
_ISSUE_TYPES = ["billing", "technical", "account", "feature-request", "security"]
_PRIORITIES  = ["low", "medium", "high", "critical"]
_T_STATUSES  = ["open", "resolved", "closed", "escalated"]


def seed_database() -> None:
    """Create and seed nl2sql_arena.db with deterministic synthetic data."""
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    # ── Customers (500 rows) ──────────────────────────────────────────────────
    customers: list[tuple] = []
    for cid in range(1, 501):
        email = fake.email() if random.random() > 0.05 else None
        signup = fake.date_between(start_date="-4y", end_date="-1y").isoformat()
        customers.append((
            cid,
            fake.name(),
            email,
            random.choice(_COUNTRIES),
            random.choice(_SEGMENTS),
            signup,
        ))
    cursor.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    # ── Products (50 rows) ───────────────────────────────────────────────────
    products: list[tuple] = []
    for pid in range(1, 51):
        price = round(random.uniform(49.99, 4999.99), 2)
        cost  = round(price * random.uniform(0.25, 0.65), 2)
        products.append((pid, _PRODUCT_NAMES[pid - 1], random.choice(_CATEGORIES), price, cost))
    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    # Build price lookup for revenue computation
    price_map = {p[0]: p[3] for p in products}

    # ── Orders (1 000 rows, good APAC/2023 coverage) ─────────────────────────
    orders: list[tuple] = []
    for oid in range(1, 1001):
        cid = random.randint(1, 500)
        pid = random.randint(1, 50)
        qty = random.randint(1, 20)
        revenue = round(price_map[pid] * qty * random.uniform(0.85, 1.15), 2)

        year = random.choices([2022, 2023, 2024], weights=[0.30, 0.40, 0.30])[0]
        order_date = fake.date_between(
            start_date=date(year, 1, 1), end_date=date(year, 12, 31)
        ).isoformat()

        region = random.choices(_REGIONS, weights=[0.25, 0.30, 0.30, 0.15])[0]
        status = random.choices(_STATUSES, weights=[0.70, 0.15, 0.10, 0.05])[0]
        orders.append((oid, cid, pid, qty, revenue, order_date, region, status))
    cursor.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)

    # ── Support Tickets (500 rows, ~20 % high-priority with resolutions) ─────
    tickets: list[tuple] = []
    for tid in range(1, 501):
        cid       = random.randint(1, 500)
        issue     = random.choice(_ISSUE_TYPES)
        priority  = random.choices(_PRIORITIES, weights=[0.30, 0.40, 0.20, 0.10])[0]
        status    = random.choices(_T_STATUSES, weights=[0.20, 0.50, 0.20, 0.10])[0]
        created   = fake.date_time_between(start_date="-3y", end_date="-1d")

        if status in ("resolved", "closed"):
            hours_to_resolve = random.uniform(0.5, 168.0)
            resolved = created + timedelta(hours=hours_to_resolve)
            resolved_at = resolved.strftime("%Y-%m-%d %H:%M:%S")
        else:
            resolved_at = None

        tickets.append((
            tid, cid, issue, priority, status,
            created.strftime("%Y-%m-%d %H:%M:%S"),
            resolved_at,
        ))
    cursor.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", tickets)

    conn.commit()
    conn.close()

    print(f"Database seeded: {DB_PATH}")
    print(f"  customers      : {len(customers)}")
    print(f"  products       : {len(products)}")
    print(f"  orders         : {len(orders)}")
    print(f"  support_tickets: {len(tickets)}")


if __name__ == "__main__":
    seed_database()
