#!/usr/bin/env bash
set -euo pipefail

# Run DARP vs RAO* using the paper-derived experiment configuration.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/DARP-vs-RAOstar}"

export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  VENV_DIR="${VENV_DIR}" bash "${ROOT_DIR}/tools/install.sh"
fi

mkdir -p "${OUTPUT_DIR}"

RUN_ARGS=(
  --trials "${TRIALS:-25}"
  --output "${OUTPUT_DIR}/raw.csv"
  --summary "${OUTPUT_DIR}/summary.md"
  --resume
)

if [[ -n "${CONSTRAINED_POMDP_REPO:-}" ]]; then
  RUN_ARGS+=(--constrained-pomdp-repo "${CONSTRAINED_POMDP_REPO}")
fi
if [[ -n "${RAOSTAR_CHECKOUT:-}" ]]; then
  RUN_ARGS+=(--raostar-checkout "${RAOSTAR_CHECKOUT}")
fi
if [[ -n "${BASELINE_CACHE:-}" ]]; then
  RUN_ARGS+=(--baseline-cache "${BASELINE_CACHE}")
fi

"${VENV_DIR}/bin/python" -m experiments.DARP-vs-RAOstar.run "${RUN_ARGS[@]}"
