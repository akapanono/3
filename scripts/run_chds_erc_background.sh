#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_chds_erc_background.sh
#
# Optional environment variables:
#   DATASET_DIR=data/IEMOCAP
#   MODEL_NAME_OR_PATH=princeton-nlp/sup-simcse-roberta-large
#   OUTPUT_DIR=outputs/chds_erc_iemocap
#   CUDA_VISIBLE_DEVICES=0
#   EXTRA_ARGS="--epochs 8 --batch_size 8"

DATASET_DIR="${DATASET_DIR:-data/IEMOCAP}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-princeton-nlp/sup-simcse-roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/chds_erc_iemocap}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

nohup python -u -m chds_erc.train \
  --dataset_dir "${DATASET_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS} \
  > "${OUTPUT_DIR}/stdout.log" 2>&1 &

PID="$!"
echo "${PID}" > "${OUTPUT_DIR}/train.pid"

echo "CHDS-ERC training started in background."
echo "PID: ${PID}"
echo "stdout: ${OUTPUT_DIR}/stdout.log"
echo "epoch text log: ${OUTPUT_DIR}/epoch_results.txt"
echo "epoch csv: ${OUTPUT_DIR}/epoch_metrics.csv"
echo "jsonl metrics: ${OUTPUT_DIR}/metrics.jsonl"
echo
echo "Watch progress:"
echo "  tail -f ${OUTPUT_DIR}/epoch_results.txt"
echo "  tail -f ${OUTPUT_DIR}/stdout.log"

