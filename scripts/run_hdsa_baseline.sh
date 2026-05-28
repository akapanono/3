#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u src/run.py \
  --use_hdsa \
  --dataset_name "${DATASET_NAME:-IEMOCAP}" \
  --dataset_dir "${DATASET_DIR:-data/IEMOCAP}" \
  --bert_path "${BERT_PATH:-pretrained/sup-simcse-roberta-large}" \
  --domain_anchor_path "${DOMAIN_ANCHOR_PATH:-domain_anchors/IEMOCAP_M3_hyp.pt}" \
  --output_dir "${OUTPUT_DIR:-outputs/hdsa_erc_iemocap_baseline}" \
  --experiment_name "${EXPERIMENT_NAME:-exp0_restore_baseline}" \
  --local_files_only \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --anchor_temperature 0.1 \
  --proto_temperature 0.1 \
  --anchor_logit_weight 0.3 \
  --proto_loss_weight 0.5 \
  --compact_loss_weight 0.1 \
  --center_weight 0.1 \
  --div_weight 1.0 \
  --preserve_weight 0.5 \
  --same_upper 0.90 \
  --ot_epsilon 0.02 \
  --ot_iters 50 \
  --ot_sharpen_power 2.0 \
  --prototype_momentum 0.95 \
  --ema_conf_threshold 0.45 \
  --early_stop \
  --early_stop_metric dev_weighted_f1 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0001 \
  --save_best_dev \
  --save_best_test \
  "$@"
