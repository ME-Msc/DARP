#!/usr/bin/env bash
set -euo pipefail

# Install DARP's locked Python environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
echo "DARP root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Virtual environment: ${VENV_DIR}"

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 12, 3) else 1)'; then
  echo "DARP's locked artifact requires CPython 3.12.3." >&2
  exit 2
fi

echo "Creating/updating DARP virtual environment..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-lock.txt"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-deps -e "${ROOT_DIR}"

cat <<EOF

Installation finished.

Activate DARP:
  source "${VENV_DIR}/bin/activate"

Both planners require a valid local Gurobi license.
EOF
