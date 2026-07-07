#!/usr/bin/env bash
set -euo pipefail

# Install DARP's Python environment.
# 安装 DARP Python 环境。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

echo "DARP root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Virtual environment: ${VENV_DIR}"

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
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[gurobi]" -r "${ROOT_DIR}/requirements-dev.txt"

cat <<EOF

Installation finished.

Activate DARP:
  source "${VENV_DIR}/bin/activate"

Gurobi note:
  gurobipy is installed through the DARP extra, but a valid Gurobi license is
  still required for full-ilp/hilp experiments.
EOF
