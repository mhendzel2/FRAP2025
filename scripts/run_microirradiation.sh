#!/usr/bin/env bash
set -euo pipefail

# Laser Microirradiation Analysis - Standalone Streamlit launcher
# This script launches the separate microirradiation app without affecting the main FRAP app.

usage() {
  cat <<'USAGE'
Usage: scripts/run_microirradiation.sh [--port PORT] [--host HOST]

Options:
  --port PORT   Port to run the app on (default: 5001)
  --host HOST   Interface/address to bind (default: 127.0.0.1). Use 0.0.0.0 to listen on all interfaces.

Examples:
  scripts/run_microirradiation.sh
  scripts/run_microirradiation.sh --port 5002
  scripts/run_microirradiation.sh --host 0.0.0.0 --port 5001

Notes:
- This is a separate app. It does NOT modify or replace the main FRAP Streamlit app.
- You can run both apps side-by-side by using different ports.
USAGE
}

PORT=${PORT:-5001}
HOST=${HOST:-127.0.0.1}

# Parse simple flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2;;
    --host)
      HOST="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 1;;
  esac
done

# Resolve repo root and app path
SCRIPT_DIR="$(cd -- ""$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$REPO_ROOT/streamlit_microirradiation.py"

if [[ ! -f "$APP_PATH" ]]; then
  echo "Error: streamlit_microirradiation.py not found at $APP_PATH" >&2
  exit 1
fi

# Ensure Streamlit is available
if ! command -v streamlit >/dev/null 2>&1; then
  echo "Streamlit is not installed or not on PATH." >&2
  echo "Install with one of the following:" >&2
  echo "  pip install -r requirements.txt" >&2
  echo "  or: pip install streamlit numpy scipy pandas scikit-image matplotlib opencv-python" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Launching Laser Microirradiation app on $HOST:$PORT ..."
exec streamlit run "$APP_PATH" \
  --server.port "$PORT" \
  --server.address "$HOST"