#!/usr/bin/env bash
set -euo pipefail

# Run Table 2 as a named batch; reporting is separate.
# 命名批次运行 Table 2，对比表格独立生成。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
cd "${ROOT_DIR}"
RUN_ARGS=(--config experiments/grid/configs/table2.json --name "${RUN_NAME:-table2-new}" --trials "${TRIALS:-1}")
if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi
if [[ -n "${CONSTRAINED_POMDP_REPO:-}" ]]; then
  RUN_ARGS+=(--constrained-repo "${CONSTRAINED_POMDP_REPO}")
fi
if [[ -n "${RAOSTAR_CHECKOUT:-}" ]]; then
  RUN_ARGS+=(--raostar-repo "${RAOSTAR_CHECKOUT}")
fi
"${VENV_DIR}/bin/python" experiments/run.py "${RUN_ARGS[@]}" "$@"
