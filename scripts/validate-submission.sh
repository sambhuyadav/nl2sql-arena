#!/usr/bin/env bash
# validate-submission.sh — OpenEnv Submission Validator
#
# Prerequisites:
#   Docker, openenv-core (pip install openenv-core), curl
#
# Usage:
#   ./scripts/validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"
PING_URL="${PING_URL%/}"
PASS=0

pass() { echo "PASSED -- $1"; PASS=$((PASS + 1)); }
fail() { echo "FAILED -- $1"; }
stop_at() { echo "Validation stopped at $1. Fix above and retry."; exit 1; }

echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"
echo "Repo:     $REPO_DIR"
echo "Ping URL: $PING_URL"
echo ""

# ── Step 1: Ping HF Space ────────────────────────────────────────────────────
echo "Step 1/3: Pinging HF Space ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

# ── Step 2: Docker Build ──────────────────────────────────────────────────────
echo "Step 2/3: Running docker build ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found — install Docker first"
  stop_at "Step 2"
fi

if   [ -f "$REPO_DIR/Dockerfile" ];        then DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

BUILD_OK=false
if command -v timeout &>/dev/null; then
  timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" && BUILD_OK=true
elif command -v gtimeout &>/dev/null; then
  gtimeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" && BUILD_OK=true
else
  # No timeout utility (e.g. macOS without coreutils) — run without timeout
  docker build "$DOCKER_CONTEXT" && BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  stop_at "Step 2"
fi

# ── Step 3: openenv validate ──────────────────────────────────────────────────
echo "Step 3/3: Running openenv validate ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv not found — run: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
cd "$REPO_DIR" && openenv validate && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  stop_at "Step 3"
fi

echo ""
echo "========================================"
echo "  All 3/3 checks passed!"
echo "  Your submission is ready to submit."
echo "========================================"
exit 0
