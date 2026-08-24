#!/usr/bin/env bash
set -euo pipefail

# Install DARP's Python environment.
# 安装 DARP Python 环境。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
USE_LOCKFILE="${USE_LOCKFILE:-1}"
TASK_CACHE_DIR="${DARP_CACHE_DIR:-${ROOT_DIR}/.cache}"

export XDG_CACHE_HOME="${TASK_CACHE_DIR}"
export MPLCONFIGDIR="${TASK_CACHE_DIR}/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

echo "DARP root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Virtual environment: ${VENV_DIR}"

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "DARP's locked environment requires Python >= 3.12 (tested with 3.13)." >&2
  exit 2
fi

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  echo "Installing Ubuntu/Debian system packages with apt..."
  sudo apt-get update
  sudo apt-get install -y \
    git \
    python3-venv \
    python3-pip \
    build-essential \
    g++ \
    cmake
else
  echo "Skipping apt packages. Set INSTALL_SYSTEM_DEPS=1 to install them."
fi

echo "Creating/updating DARP virtual environment..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ "${USE_LOCKFILE}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-lock.txt"
  "${VENV_DIR}/bin/python" -m pip install --no-deps -e "${ROOT_DIR}"
else
  "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[gurobi]" -r "${ROOT_DIR}/requirements-dev.txt"
fi

cat <<EOF

Installation finished.

Activate DARP:
  source "${VENV_DIR}/bin/activate"

Gurobi note:
  gurobipy is installed through the DARP extra, but a valid Gurobi license is
   still required for full-ilp/hilp experiments.
EOF
