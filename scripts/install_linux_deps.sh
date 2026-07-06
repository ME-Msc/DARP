#!/usr/bin/env bash
set -euo pipefail

# Install DARP's Python environment and, optionally, PROST/rddlsim dependencies.
# 安装 DARP Python 环境，并可选安装 PROST/rddlsim 依赖。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
INSTALL_RDDLSIM="${INSTALL_RDDLSIM:-1}"
INSTALL_PROST="${INSTALL_PROST:-1}"
BUILD_RDDLSIM="${BUILD_RDDLSIM:-1}"
BUILD_PROST="${BUILD_PROST:-1}"

PROST_ROOT="${PROST_ROOT:-${ROOT_DIR}/../prost-planner}"
PROST_REPO_URL="${PROST_REPO_URL:-https://github.com/prost-planner/prost.git}"
RDDLSIM_ROOT="${RDDLSIM_ROOT:-${ROOT_DIR}/../rddlsim}"
RDDLSIM_REPO_URL="${RDDLSIM_REPO_URL:-https://github.com/ssanner/rddlsim.git}"

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
    cmake \
    bison \
    flex \
    libbdd-dev \
    z3 \
    libz3-dev \
    default-jre \
    default-jdk
else
  echo "Skipping apt packages. Set INSTALL_SYSTEM_DEPS=1 to install them."
fi

echo "Creating/updating DARP virtual environment..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[gurobi]" -r "${ROOT_DIR}/requirements-dev.txt"

if [[ "${INSTALL_RDDLSIM}" == "1" ]]; then
  if [[ ! -d "${RDDLSIM_ROOT}/.git" ]]; then
    echo "Cloning rddlsim into ${RDDLSIM_ROOT}..."
    git clone "${RDDLSIM_REPO_URL}" "${RDDLSIM_ROOT}"
  else
    echo "Using existing rddlsim checkout: ${RDDLSIM_ROOT}"
  fi
  if [[ "${BUILD_RDDLSIM}" == "1" ]]; then
    echo "Building rddlsim..."
    (cd "${RDDLSIM_ROOT}" && ./compile)
  fi
else
  echo "Skipping rddlsim clone/build."
fi

if [[ "${INSTALL_PROST}" == "1" ]]; then
  if [[ ! -d "${PROST_ROOT}/.git" ]]; then
    echo "Cloning PROST into ${PROST_ROOT}..."
    git clone "${PROST_REPO_URL}" "${PROST_ROOT}"
  else
    echo "Using existing PROST checkout: ${PROST_ROOT}"
  fi
  if [[ "${BUILD_PROST}" == "1" ]]; then
    echo "Building PROST release binaries..."
    (cd "${PROST_ROOT}" && ./build.py)
  fi
else
  echo "Skipping PROST clone/build."
fi

cat <<EOF

Installation finished.

Activate DARP:
  source "${VENV_DIR}/bin/activate"

Recommended environment for DARP/PROST comparisons:
  export PROST_ROOT="${PROST_ROOT}"
  export RDDLSIM_ROOT="${RDDLSIM_ROOT}"
  export PROST_PYTHON="${VENV_DIR}/bin/python"

Gurobi note:
  gurobipy is installed through the DARP extra, but a valid Gurobi license is
  still required for full-ilp/hilp experiments.
EOF
