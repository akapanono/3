#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --dataset_dir data/IEMOCAP \
  --bert_path pretrained/sup-simcse-roberta-large \
  --domain_anchor_path domain_anchors/IEMOCAP_M3_hyp.pt \
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
  --class_adaptive_ema \
  --default_ema_conf_threshold 0.45 \
  --low_conf_classes angry,frustrated \
  --low_conf_threshold 0.38 \
  --happy_conf_threshold 0.42 \
  --use_ema_fallback \
  --fallback_momentum 0.98 \
  --pair_loss_weight 0.1 \
  --pair_margin 0.3 \
  --pair_anchor_loss_weight 0.05 \
  --pair_anchor_upper 0.20 \
  --happy_ce_weight 1.3 \
  --use_intensity_head \
  --intensity_loss_weight 0.05 \
  "$@"
