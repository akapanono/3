#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-experiments/sweep_configs/base_iemocap.json}"
SWEEP_CONFIG="${SWEEP_CONFIG:-experiments/sweep_configs/sweep_test_best_minimal.json}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs_test_best}"
SEEDS="${SEEDS:-42}"
TOP_K="${TOP_K:-5}"
MAX_RUNS="${MAX_RUNS:-}"
EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:-}"

mkdir -p "${OUTPUT_DIR}/results"

CMD=(
  python -u experiments/sweep.py
  --base_config "${BASE_CONFIG}"
  --sweep_config "${SWEEP_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --seeds "${SEEDS}"
  --top_k "${TOP_K}"
)

if [[ -n "${MAX_RUNS}" ]]; then
  CMD+=(--max_runs "${MAX_RUNS}")
fi

nohup "${CMD[@]}" ${EXTRA_SWEEP_ARGS} \
  > "${OUTPUT_DIR}/sweep_stdout.log" 2>&1 &

PID="$!"
echo "${PID}" > "${OUTPUT_DIR}/sweep.pid"
echo "HDSA-ERC sweep started."
echo "PID: ${PID}"
echo "Watch: tail -f ${OUTPUT_DIR}/sweep_stdout.log"
echo "Results: ${OUTPUT_DIR}/results/sweep_results.csv"
