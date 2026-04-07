FROM python:3.11.9-slim-bookworm

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Seed the database at build time so ground truths are deterministic ────────
RUN python database.py

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# Hugging Face Spaces requires port 7860
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
