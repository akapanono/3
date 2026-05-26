#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-IEMOCAP}"
DATASET_DIR="${DATASET_DIR:-data/${DATASET_NAME}}"
BERT_PATH="${BERT_PATH:-pretrained/sup-simcse-roberta-large}"
DOMAIN_ANCHOR_PATH="${DOMAIN_ANCHOR_PATH:-domain_anchors/${DATASET_NAME}_M3_hyp.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hdsa_erc_${DATASET_NAME}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

nohup python -u src/run.py \
  --use_hdsa \
  --dataset_name "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --bert_path "${BERT_PATH}" \
  --domain_anchor_path "${DOMAIN_ANCHOR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS} \
  > "${OUTPUT_DIR}/stdout.log" 2>&1 &

PID="$!"
echo "${PID}" > "${OUTPUT_DIR}/train.pid"
echo "HDSA-ERC training started."
echo "PID: ${PID}"
echo "Watch: tail -f ${OUTPUT_DIR}/epoch_results.txt"
echo "Stdout: tail -f ${OUTPUT_DIR}/stdout.log"

