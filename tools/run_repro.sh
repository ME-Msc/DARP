#!/usr/bin/env bash
set -euo pipefail

# Run DARP vs RAO* using the paper-derived experiment configuration.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/DARP-vs-RAOstar-grid}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
cd "${ROOT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  VENV_DIR="${VENV_DIR}" bash "${ROOT_DIR}/tools/install.sh"
fi

mkdir -p "${OUTPUT_DIR}"

RUN_ARGS=(
  --trials "${TRIALS:-25}"
  --output "${OUTPUT_DIR}/table2-raw.csv"
  --summary "${OUTPUT_DIR}/table2.md"
)

if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi
if [[ -n "${CONSTRAINED_POMDP_REPO:-}" ]]; then
  RUN_ARGS+=(--constrained-pomdp-repo "${CONSTRAINED_POMDP_REPO}")
fi
if [[ -n "${RAOSTAR_CHECKOUT:-}" ]]; then
  RUN_ARGS+=(--raostar-checkout "${RAOSTAR_CHECKOUT}")
fi
if [[ -n "${BASELINE_CACHE:-}" ]]; then
  RUN_ARGS+=(--baseline-cache "${BASELINE_CACHE}")
fi

"${VENV_DIR}/bin/python" -m experiments.DARP-vs-RAOstar-grid.run "${RUN_ARGS[@]}"
