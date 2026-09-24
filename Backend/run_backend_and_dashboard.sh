#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/../venv/bin/python}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

require_module() {
  local module_name="$1"
  if ! "$VENV_PYTHON" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "Missing Python module: ${module_name}"
    echo "Install it with: $VENV_PYTHON -m pip install ${module_name}"
    exit 1
  fi
}

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python executable not found: $VENV_PYTHON"
  echo "Set VENV_PYTHON to your interpreter path and retry."
  exit 1
fi

require_module streamlit

if port_in_use "$API_PORT"; then
  echo "Port $API_PORT is already in use."
  lsof -nP -iTCP:"$API_PORT" -sTCP:LISTEN || true
  echo "Free it first, for example: lsof -ti tcp:$API_PORT | xargs kill -9"
  exit 1
fi

if port_in_use "$DASHBOARD_PORT"; then
  echo "Port $DASHBOARD_PORT is already in use."
  lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN || true
  echo "Use another dashboard port, e.g. DASHBOARD_PORT=8502 ./run_backend_and_dashboard.sh"
  exit 1
fi

cd "$ROOT_DIR"

api_pid=""
dash_pid=""

cleanup() {
  echo
  echo "Stopping services..."
  if [[ -n "$api_pid" ]] && kill -0 "$api_pid" 2>/dev/null; then
    kill "$api_pid" 2>/dev/null || true
  fi
  if [[ -n "$dash_pid" ]] && kill -0 "$dash_pid" 2>/dev/null; then
    kill "$dash_pid" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "Starting backend API on http://$API_HOST:$API_PORT"
"$VENV_PYTHON" -m uvicorn main:app --host "$API_HOST" --port "$API_PORT" --reload &
api_pid=$!

echo "Starting dashboard on http://127.0.0.1:$DASHBOARD_PORT"
"$VENV_PYTHON" -m streamlit run dashboard_app.py --server.port "$DASHBOARD_PORT" --server.headless true &
dash_pid=$!

echo
echo "Services started:"
echo "  API:       http://$API_HOST:$API_PORT"
echo "  Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
echo "Press Ctrl+C to stop both."
echo

wait
