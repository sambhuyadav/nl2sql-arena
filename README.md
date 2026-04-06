# NL2SQL Arena

> **An interactive RL arena for training agents to reason over structured data — from natural language to SQL, through a structured Analysis DSL.**

---

## 1. Why NL2SQL Arena?

Most NL2SQL benchmarks (Spider, BIRD, WikiSQL) are **static evaluation datasets**: you submit a SQL string, compare to a gold answer, done. They offer no:

- **Intermediate reasoning signal** — an agent that writes perfect SQL in one shot is treated identically to one that needed 8 corrections.
- **Structured intermediate language** — agents must go straight from English to SQL, skipping the planning step a human analyst would use.
- **Adversarial challenges** — no broken queries to debug, no deliberately tricky schema hints.
- **Shaped rewards for RL training** — binary pass/fail makes reinforcement learning slow and unstable.

**NL2SQL Arena** is different:

| Feature | Static Benchmarks | NL2SQL Arena |
|---|---|---|
| Reward signal | Binary (0 or 1) | Shaped (0.0–1.0, per-component) |
| Intermediate language | None | Analysis DSL |
| Adversarial tasks | No | Yes — broken DSL to debug |
| RL-ready | No | Yes — step-based, multi-turn |
| Real business questions | Varies | Yes — revenue, customers, tickets |
| Self-correction loop | No | Yes — errors fed back as observations |

---

## 2. Environment Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                              │
│                                                                │
│  ┌──────────┐    ArenaObservation     ┌─────────────────────┐  │
│  │          │◄────────────────────────│  NL2SQL Arena Env   │  │
│  │  Agent   │                         │  (FastAPI server)   │  │
│  │  (LLM)   │─────ArenaAction────────►│                     │  │
│  │          │    {dsl, explain}        │  1. Parse DSL       │  │
│  └──────────┘                         │  2. Compile → SQL   │  │
│                                       │  3. Execute SQL     │  │
│  Observation fields:                  │  4. Grade result    │  │
│  • question (NL)                      │  5. Compute reward  │  │
│  • schema_hint                        │                     │  │
│  • broken_dsl (Task 3)                └─────────────────────┘  │
│  • last_sql_executed                          │                 │
│  • last_result_preview                   ArenaReward           │
│  • last_error                         {value, breakdown, msg}  │
│  • step_count / done                                           │
└────────────────────────────────────────────────────────────────┘
```

The agent receives a natural language business question, writes an **Analysis DSL** program, and the environment compiles it to SQL, executes it against a live SQLite database, grades the result, and returns a shaped reward with a detailed breakdown.

---

## 3. The Analysis DSL

The Analysis DSL is a structured intermediate language that mirrors how a human analyst thinks about a query before writing SQL.

### Grammar

```
QUERY <table>
  [WHERE  <col> <op> <val> [AND <col> <op> <val> ...]]
  [JOIN   <table2> ON <col1> = <col2>]
  [AGGREGATE <fn>(<col>[, <col2>]) AS <alias> [BY <group_col>[, ...]]]
  [SORT   <col> [ASC|DESC]]
  [LIMIT  <n>]
  [COMPARE previous_period]
  [EXPLAIN <free-text reasoning>]
```

**Aggregate functions:** `sum`, `avg`, `count`, `min`, `max`, `count_distinct`, `avg_hours`

**Special:** `avg_hours(col1, col2)` computes `AVG((julianday(col1) - julianday(col2)) * 24)` — average elapsed hours between two datetime columns.

**WHERE operators:** `=`, `!=`, `>`, `<`, `>=`, `<=`, `BETWEEN`, `LIKE`, `IN`, `IS NULL`, `IS NOT NULL`

### Example 1 — Simple filter + aggregate

```
QUERY orders
  WHERE region = "APAC" AND order_date BETWEEN "2023-01-01" AND "2023-12-31"
  AGGREGATE sum(revenue) AS total_revenue
  EXPLAIN Summing all APAC orders placed in calendar year 2023
```

Compiles to:
```sql
SELECT SUM(revenue) AS total_revenue
FROM orders
WHERE region = 'APAC' AND order_date BETWEEN '2023-01-01' AND '2023-12-31'
```

### Example 2 — JOIN + GROUP BY + SORT + LIMIT

```
QUERY orders
  JOIN customers ON orders.customer_id = customers.customer_id
  AGGREGATE sum(revenue) AS total_revenue BY customers.name, customers.country
  SORT total_revenue DESC
  LIMIT 5
  EXPLAIN Joining orders with customers to rank top 5 by spend
```

Compiles to:
```sql
SELECT customers.name, customers.country, SUM(revenue) AS total_revenue
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id
GROUP BY customers.name, customers.country
ORDER BY total_revenue DESC
LIMIT 5
```

### Example 3 — Time-difference aggregation (Task 3 pattern)

```
QUERY support_tickets
  WHERE priority = "high" AND resolved_at IS NOT NULL
  AGGREGATE avg_hours(resolved_at, created_at) AS avg_resolution_hours BY issue_type
  SORT avg_resolution_hours DESC
  EXPLAIN Average hours to resolve high-priority tickets, broken down by issue type
```

Compiles to:
```sql
SELECT issue_type, AVG((julianday(resolved_at) - julianday(created_at)) * 24) AS avg_resolution_hours
FROM support_tickets
WHERE priority = 'high' AND resolved_at IS NOT NULL
GROUP BY issue_type
ORDER BY avg_resolution_hours DESC
```

---

## 4. Action Space

The agent submits an `ArenaAction`:

```python
class ArenaAction(BaseModel):
    dsl: str            # Required — the full DSL program text
    explain: Optional[str] = None  # Optional — explanation (earns +0.05 bonus reward)
```

The `EXPLAIN` clause may appear inside the DSL *or* as a separate `explain` field — either earns the bonus.

---

## 5. Observation Space

```python
class ArenaObservation(BaseModel):
    task_id: str                        # 'simple-lookup' | 'multi-table-join' | 'debug-and-fix'
    question: str                       # Natural language business question
    schema_hint: str                    # Table schemas + enum values + FK info
    broken_dsl: Optional[str]           # Task 3 only — buggy DSL with 2 deliberate errors
    last_sql_executed: Optional[str]    # SQL compiled and executed in the previous step
    last_result_preview: Optional[str]  # First 3 rows of the last result set
    last_error: Optional[str]           # DSL parse error or SQL execution error
    step_count: int                     # Steps taken so far (0 on reset)
    done: bool                          # True when episode has ended
```

The `last_error` field is critical for self-correction: if the agent's SQL fails, the exact error message is returned in the next observation.

---

## 6. Task Descriptions

### Task 1 — `simple-lookup` (Easy, max 5 steps)

**Business question:**
> "What is the total revenue for the APAC region in 2023?"

**Expected DSL pattern:**
```
QUERY orders
  WHERE region = "APAC" AND order_date BETWEEN "2023-01-01" AND "2023-12-31"
  AGGREGATE sum(revenue) AS total_revenue
```

**Grader:** Compares the aggregated value to ground truth within 1% tolerance.

| Component | Reward |
|---|---|
| Valid DSL syntax | +0.05 |
| Correct table (orders) | +0.10 |
| Correct WHERE (APAC + 2023) | +0.15 |
| Correct aggregation (sum revenue) | +0.20 |
| Result matches ground truth | +0.25 |

---

### Task 2 — `multi-table-join` (Medium, max 8 steps)

**Business question:**
> "List the top 5 customers by total revenue, showing their name, country, and total spend."

**Expected DSL pattern:**
```
QUERY orders
  JOIN customers ON orders.customer_id = customers.customer_id
  AGGREGATE sum(revenue) AS total_revenue BY customers.name, customers.country
  SORT total_revenue DESC
  LIMIT 5
```

**Grader:** Compares result rows positionally (name at each of 5 positions). Full score requires correct names in correct order.

| Component | Reward |
|---|---|
| Valid DSL syntax | +0.05 |
| Correct JOIN with customer_id | +0.10 |
| Correct aggregation (sum revenue by customer) | +0.15 |
| SORT DESC + LIMIT 5 | +0.10 |
| Result matches ground truth (top-5 ordered) | +0.25 |

---

### Task 3 — `debug-and-fix` (Hard, max 10 steps)

**Business question:**
> "Find the average resolution time in hours for high-priority support tickets, grouped by issue type."

**Broken DSL presented to the agent** (contains 2 bugs):
```
QUERY support_tickets
  WHERE priority = "high"
  AGGREGATE avg(resolution_time) AS avg_resolution
  SORT avg_resolution DESC
```

**Bugs:**
1. `resolution_time` — column does not exist (should compute from `resolved_at` and `created_at`)
2. Missing `BY issue_type` — GROUP BY is absent

**Expected corrected DSL:**
```
QUERY support_tickets
  WHERE priority = "high" AND resolved_at IS NOT NULL
  AGGREGATE avg_hours(resolved_at, created_at) AS avg_resolution_hours BY issue_type
  SORT avg_resolution_hours DESC
  EXPLAIN Fixed: replaced non-existent column with avg_hours(), added BY issue_type
```

| Component | Reward |
|---|---|
| Valid DSL syntax | +0.05 |
| Correct columns (uses avg_hours with datetime cols) | +0.15 |
| Correct GROUP BY issue_type | +0.15 |
| Correct WHERE (priority=high) | +0.10 |
| Result matches ground truth | +0.25 |

**Additional penalty:** -0.10 for re-submitting the same broken DSL unchanged.

---

## 7. Reward Function

All rewards are **shaped** (not sparse) and clamped to **[0.0, 1.0]**.

### Positive Components

| Component | Weight | Condition |
|---|---|---|
| Syntax valid | +0.05 | DSL parsed without error |
| Table selection | +0.10 | Primary table matches task expectation |
| WHERE conditions | +0.15 | Filter columns/values match task |
| Aggregation | +0.20 | Correct function and column |
| Result match | +0.25 | Execution result matches ground truth |
| EXPLAIN bonus | +0.05 | EXPLAIN clause present and non-empty |

### Penalties

| Penalty | Amount | Condition |
|---|---|---|
| SQL execution error | -0.20 | Query fails in SQLite |
| Identical re-submission | -0.10 | Same DSL as previous step |
| Step efficiency | -0.05×(step−3) | Each step beyond step 3 |

### Example Reward Breakdown

```json
{
  "syntax": 0.05,
  "table": 0.10,
  "where": 0.15,
  "aggregation": 0.20,
  "result_match": 0.25,
  "explain_bonus": 0.05,
  "step_penalty": -0.10,
  "total_raw": 0.70,
  "total": 0.70
}
```

---

## 8. Database Schema

### `orders`
| Column | Type | Notes |
|---|---|---|
| order_id | INTEGER PK | |
| customer_id | INTEGER FK | → customers |
| product_id | INTEGER FK | → products |
| quantity | INTEGER | 1–20 |
| revenue | REAL | USD, may differ from unit_price × qty due to discounts |
| order_date | TEXT | ISO 8601 date, 2022–2024 |
| region | TEXT | APAC, EMEA, AMER, LATAM |
| status | TEXT | completed, pending, cancelled, refunded |

### `customers`
| Column | Type | Notes |
|---|---|---|
| customer_id | INTEGER PK | |
| name | TEXT | Faker-generated, may have duplicates |
| email | TEXT | ~5 % NULL |
| country | TEXT | USA, UK, Germany, France, Japan, Australia, Canada, India, Brazil, Singapore |
| segment | TEXT | Enterprise, SMB, Startup, Government, Education |
| signup_date | TEXT | ISO 8601 date |

### `products`
| Column | Type | Notes |
|---|---|---|
| product_id | INTEGER PK | |
| name | TEXT | 50 named B2B software products |
| category | TEXT | Software, Hardware, Services, Cloud, Security, Analytics |
| unit_price | REAL | $49.99–$4,999.99 |
| cost | REAL | 25–65 % of unit_price |

### `support_tickets`
| Column | Type | Notes |
|---|---|---|
| ticket_id | INTEGER PK | |
| customer_id | INTEGER FK | → customers |
| issue_type | TEXT | billing, technical, account, feature-request, security |
| priority | TEXT | low, medium, high, critical |
| status | TEXT | open, resolved, closed, escalated |
| created_at | TEXT | ISO 8601 datetime |
| resolved_at | TEXT | ISO 8601 datetime, NULL if unresolved |

**Sample rows (orders):**
```
order_id | customer_id | product_id | quantity | revenue   | order_date | region | status
---------|-------------|------------|----------|-----------|------------|--------|----------
1        | 42          | 7          | 3        | 4521.33   | 2023-04-15 | APAC   | completed
2        | 17          | 23         | 1        | 899.99    | 2023-07-02 | EMEA   | completed
3        | 201         | 11         | 8        | 12403.20  | 2022-11-30 | AMER   | cancelled
```

---

## 9. Quick Start

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Seed the database (required before starting the server)
python database.py

# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

The server is now available at `http://localhost:7860`.  
Swagger UI: `http://localhost:7860/docs`

**Quick smoke test:**
```bash
# Reset (start a new episode)
curl -s -X POST "http://localhost:7860/reset?task_id=simple-lookup" | python -m json.tool

# Submit a DSL step
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "dsl": "QUERY orders\n  WHERE region = \"APAC\" AND order_date BETWEEN \"2023-01-01\" AND \"2023-12-31\"\n  AGGREGATE sum(revenue) AS total_revenue\n  EXPLAIN Summing APAC 2023 orders"
    }
  }' | python -m json.tool
```

### Docker

```bash
# Build (seeds the database during build)
docker build -t nl2sql-arena .

# Run
docker run -p 7860:7860 nl2sql-arena

# Health check
curl http://localhost:7860/health
```

### Run Inference Baseline

```bash
export HF_TOKEN="your_hf_token_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

---

## 10. Baseline Scores — Qwen/Qwen2.5-72B-Instruct

Results on default seeded database (`Faker.seed(42)`), measured at `temperature=0.2`:

| Task | Avg Steps | Score | Success |
|---|---|---|---|
| simple-lookup | 1 | 0.800 | 100% |
| multi-table-join | 1 | 0.650 | 100% |
| debug-and-fix | 1 | 0.750 | 100% |
| **Overall** | **1** | **0.733** | **100%** |

*Model solves all three tasks on the first step, demonstrating that Qwen2.5-72B-Instruct can reliably parse the Analysis DSL grammar and self-correct from the broken DSL in Task 3.*

---

## 11. Example Episode

Full interaction trace for **Task 1 (simple-lookup)**:

```
POST /reset?task_id=simple-lookup
→ {
    "task_id": "simple-lookup",
    "question": "What is the total revenue for the APAC region in 2023?",
    "schema_hint": "Tables:\n- orders(order_id, customer_id, ...) ...",
    "step_count": 0,
    "done": false
  }

─── Step 1 ─────────────────────────────────────────────────────────

POST /step
body: {
  "action": {
    "dsl": "QUERY orders\n  WHERE region = \"APAC\"\n  AGGREGATE sum(revenue) AS total_revenue\n  EXPLAIN Filtering APAC orders"
  }
}
→ {
    "observation": {
      "last_sql_executed": "SELECT SUM(revenue) AS total_revenue FROM orders WHERE region = 'APAC'",
      "last_result_preview": "{'total_revenue': 4823901.44}",
      "last_error": null,
      "step_count": 1,
      "done": false
    },
    "reward": {
      "value": 0.42,
      "breakdown": {
        "syntax": 0.05, "table": 0.10, "aggregation": 0.20,
        "explain_bonus": 0.05, "result_close": 0.12,
        "where": 0.0,
        "total": 0.42
      },
      "message": "Valid DSL syntax (+0.05) | Correct table(s) selected (+0.10) | Aggregation (+0.20) | EXPLAIN clause bonus (+0.05) | Result match (+0.12, grade=0.48)"
    },
    "done": false
  }
```

*Reward is 0.42 because the WHERE clause is missing the year filter (2023). The agent's result sums all APAC years.*

```
─── Step 2 ─────────────────────────────────────────────────────────

POST /step
body: {
  "action": {
    "dsl": "QUERY orders\n  WHERE region = \"APAC\" AND order_date BETWEEN \"2023-01-01\" AND \"2023-12-31\"\n  AGGREGATE sum(revenue) AS total_revenue\n  EXPLAIN APAC revenue 2023 only"
  }
}
→ {
    "observation": {
      "last_sql_executed": "SELECT SUM(revenue) AS total_revenue FROM orders WHERE region = 'APAC' AND order_date BETWEEN '2023-01-01' AND '2023-12-31'",
      "last_result_preview": "{'total_revenue': 1247832.15}",
      "last_error": null,
      "step_count": 2,
      "done": false
    },
    "reward": {
      "value": 0.75,
      "breakdown": {
        "syntax": 0.05, "table": 0.10, "where": 0.15,
        "aggregation": 0.20, "result_match": 0.25,
        "explain_bonus": 0.05,
        "total": 0.75
      },
      "message": "Valid DSL syntax (+0.05) | Correct table(s) (+0.10) | WHERE conditions (+0.15) | Aggregation (+0.20) | Result match (+0.25) | EXPLAIN bonus (+0.05)"
    },
    "done": false
  }
```

*Agent self-corrected, added the year filter, and achieved a 0.75 reward.*

---

## 12. API Reference

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode. Query param: `task_id`. Returns `ArenaObservation`. |
| POST | `/step` | Submit action. Body: `StepRequest`. Returns `StepResponse`. |
| GET | `/state` | Current session state. Header: `X-Session-Id`. |
| GET | `/health` | Health check. Always 200. |
| GET | `/metrics` | Aggregate stats across all episodes. |

Session management: pass `X-Session-Id` header. If omitted, a new UUID is generated and returned in the response header.

---

## 13. Security

- **SELECT-only execution:** The DSL parser never produces INSERT/UPDATE/DELETE/DROP. The `execute_query()` function additionally validates and rejects any non-SELECT SQL before it reaches SQLite.
- **No eval()/exec():** The DSL parser uses regex and string composition exclusively.
- **Isolated sessions:** Each session has its own state; sessions cannot interfere with each other.
- **Parameter safety:** All string literals in WHERE clauses go through quote normalization (double→single), not string interpolation into raw SQL.

---

## 14. License

MIT License. Built for the OpenEnv Hackathon hosted by Meta and Hugging Face.
