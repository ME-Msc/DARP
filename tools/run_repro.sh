#!/usr/bin/env bash
set -euo pipefail

# Run DARP vs RAO* using the paper-derived experiment configuration.
# 按论文配置运行 DARP 与 RAO* 对照实验。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/DARP-vs-RAOstar-grid/output}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
cd "${ROOT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  VENV_DIR="${VENV_DIR}" bash "${ROOT_DIR}/tools/install.sh"
fi

mkdir -p "${OUTPUT_DIR}"

# Separate parameter sweeps from the published baseline artifacts.
# 参数扫描使用独立文件名，避免覆盖已有基线结果。
REPRO_STEM="table2"
if [[ "${RANK_ALPHA:-1}" != "1" || "${RANK_LAMBDA:-1}" != "1" ]]; then
  REPRO_STEM="table2-rank-a${RANK_ALPHA:-1}-l${RANK_LAMBDA:-1}"
fi
RUN_ARGS=(
  --trials "${TRIALS:-25}"
  --rank-alpha "${RANK_ALPHA:-1}"
  --rank-lambda "${RANK_LAMBDA:-1}"
  --output "${OUTPUT_DIR}/${REPRO_STEM}-raw.csv"
  --summary "${OUTPUT_DIR}/${REPRO_STEM}.md"
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
