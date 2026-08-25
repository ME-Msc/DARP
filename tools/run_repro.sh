#!/usr/bin/env bash
set -euo pipefail

# Run the publication comparison against the pinned upstream Quad model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/raostar-quad}"

if [[ -z "${RAOSTAR_CHECKOUT:-}" ]]; then
  echo "Set RAOSTAR_CHECKOUT to the clean manifest-pinned RAOStar checkout." >&2
  exit 2
fi
if [[ "${RAOSTAR_ACCEPT_NO_LICENSE:-}" != "1" ]]; then
  echo "Set RAOSTAR_ACCEPT_NO_LICENSE=1 after reviewing the upstream license notice." >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  VENV_DIR="${VENV_DIR}" bash "${ROOT_DIR}/tools/install.sh"
fi

mkdir -p "${OUTPUT_DIR}"
"${VENV_DIR}/bin/python" -m experiments.scripts.run_raostar_quad \
  --checkout "${RAOSTAR_CHECKOUT}" \
  --python "${RAOSTAR_PYTHON:-${VENV_DIR}/bin/python}" \
  --accept-no-license \
  --timeout "${TIMEOUT_SECONDS:-300}" \
  --repetitions "${REPETITIONS:-25}" \
  --include-full-ilp \
  --full-ilp-max-horizon 2 \
  --output "${OUTPUT_DIR}/raostar_quad_results.jsonl"
