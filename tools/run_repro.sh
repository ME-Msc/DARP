#!/usr/bin/env bash
set -euo pipefail

# Rebuild (when needed), test, and run the default HILP/RAO*-style matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/darp-review}"
TASK_CACHE_DIR="${DARP_CACHE_DIR:-${ROOT_DIR}/.cache}"

export XDG_CACHE_HOME="${TASK_CACHE_DIR}"
export MPLCONFIGDIR="${TASK_CACHE_DIR}/matplotlib"
mkdir -p "${OUTPUT_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  VENV_DIR="${VENV_DIR}" bash "${ROOT_DIR}/tools/install_linux_deps.sh"
fi

"${VENV_DIR}/bin/python" -m experiments.scripts.capture_environment \
  --repository "${ROOT_DIR}" \
  --output "${OUTPUT_DIR}/environment.json"
"${VENV_DIR}/bin/python" -m pytest -q
"${VENV_DIR}/bin/python" -m experiments.scripts.run_paper_grid \
  --horizons 3 \
  --risk-budgets 0.1 0.2 0.3 \
  --repetitions "${REPETITIONS:-3}" \
  --algorithms hilp rao-star-style \
  --expansion-rounds "${EXPANSION_ROUNDS:-100}" \
  --timeout-seconds "${TIMEOUT_SECONDS:-60}" \
  --output "${OUTPUT_DIR}/paper_grid_results.csv" \
  --summary-output "${OUTPUT_DIR}/paper_grid_summary.csv" \
  "$@"

# Exact finite-prefix cross-check. The terminal tail is disabled so HILP,
# RAO*-style exhaustive DP, and full-ILP optimize the same horizon-2 objective.
"${VENV_DIR}/bin/python" -m experiments.scripts.run_paper_grid \
  --horizons 2 \
  --risk-budgets 0.1 \
  --repetitions 1 \
  --algorithms hilp rao-star-style \
  --include-full-ilp \
  --full-ilp-max-horizon 2 \
  --disable-terminal-tail \
  --expansion-rounds "${EXPANSION_ROUNDS:-100}" \
  --timeout-seconds "${TIMEOUT_SECONDS:-60}" \
  --output "${OUTPUT_DIR}/paper_grid_oracle_h2.csv" \
  --summary-output "${OUTPUT_DIR}/paper_grid_oracle_h2_summary.csv"
