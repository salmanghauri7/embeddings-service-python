#!/usr/bin/env sh
set -eu

# Default HF port
export PORT="${PORT:-7860}"
export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
export REDIS_PORT="${REDIS_PORT:-6379}"

# Normalize accidental whitespace/newlines from platform env vars.
REDIS_HOST="$(printf '%s' "$REDIS_HOST" | tr -d '[:space:]')"
REDIS_PORT="$(printf '%s' "$REDIS_PORT" | tr -d '[:space:]')"

# In single-container mode, compose-style host 'redis' is invalid. Use localhost.
if [ "${RUN_REDIS:-1}" = "1" ] && [ "$REDIS_HOST" = "redis" ]; then
  REDIS_HOST="127.0.0.1"
fi

export REDIS_HOST REDIS_PORT

if [ "${RUN_REDIS:-1}" = "1" ]; then
  # Added --dir /tmp to ensure Redis can write its pid/temp files
  redis-server --bind "$REDIS_HOST" --port "$REDIS_PORT" --save "" --appendonly no --daemonize yes --dir /tmp
  echo "Started redis-server on ${REDIS_HOST}:${REDIS_PORT}"
fi

if [ "${RUN_WORKER:-1}" = "1" ]; then
  # Run worker in background
  python -m arq app.worker.WorkerSettings &
  echo "Started ARQ worker"
fi

# Use exec so uvicorn becomes PID 1 and receives shutdown signals correctly
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"