# HDSA-ERC

This project implements **Hyperspherical Domain Sub-anchor with Optimal Transport for ERC**.

The old CHDS implementation has been removed. The current code path is HDSA only:

1. Generate class-wise KMeans domain sub-anchors.
2. Pretrain anchors on the hypersphere with inter/domain/rank/preserve losses.
3. Train ERC with CE + OT-based prototype soft CE + compactness loss.
4. Update domain sub-anchors by high-confidence EMA according to OT assignments.
5. Run confusion-related changes as small ablations, enabled one at a time from the command line.

## 1. Generate KMeans Domain Anchors

```bash
python src/anchors/generate_domain_anchors.py \
  --dataset_name IEMOCAP \
  --dataset_dir data/IEMOCAP \
  --bert_path pretrained/sup-simcse-roberta-large \
  --local_files_only \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --output_path domain_anchors/IEMOCAP_M3_kmeans.pt
```

## 2. Pretrain Hyperspherical Anchors

```bash
python src/anchors/pretrain_hyp_domain_anchors.py \
  --init_anchor_path domain_anchors/IEMOCAP_M3_kmeans.pt \
  --output_anchor_path domain_anchors/IEMOCAP_M3_hyp.pt \
  --anchor_pretrain_epochs 1000 \
  --anchor_pretrain_lr 0.1 \
  --domain_weight 1.0 \
  --center_weight 0.1 \
  --div_weight 1.0 \
  --rank_weight 1.0 \
  --preserve_weight 0.5 \
  --same_upper 0.90
```

## 3. Train HDSA-ERC Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --dataset_dir data/IEMOCAP \
  --bert_path pretrained/sup-simcse-roberta-large \
  --domain_anchor_path domain_anchors/IEMOCAP_M3_hyp.pt \
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
  --experiment_name exp0_restore_baseline
```

Or run the baseline wrapper:

```bash
bash scripts/run_hdsa_baseline.sh
```

## Background Training

```bash
CUDA_VISIBLE_DEVICES=0 \
DATASET_NAME=IEMOCAP \
DATASET_DIR=data/IEMOCAP \
BERT_PATH=pretrained/sup-simcse-roberta-large \
DOMAIN_ANCHOR_PATH=domain_anchors/IEMOCAP_M3_hyp.pt \
OUTPUT_DIR=outputs/hdsa_erc_iemocap \
EXTRA_ARGS="--local_files_only --epochs 8 --batch_size 8 --eval_batch_size 16 --early_stop --early_stop_metric dev_weighted_f1 --early_stop_patience 3 --save_best_dev --save_best_test --experiment_name exp0_restore_baseline" \
bash scripts/run_hdsa_background.sh
```

Recommended stepwise ablations:

```bash
# Exp0: anti-collapse baseline
EXTRA_ARGS="--local_files_only --epochs 8 --batch_size 8 --eval_batch_size 16 --experiment_name exp0_baseline"

# Exp1: only class-adaptive EMA
EXTRA_ARGS="--local_files_only --epochs 8 --batch_size 8 --eval_batch_size 16 --experiment_name exp1_class_adaptive_ema --class_adaptive_ema --default_ema_conf_threshold 0.45 --low_conf_classes angry,frustrated --low_conf_threshold 0.40 --happy_conf_threshold 0.42"

# Exp2: only weak pairwise confusion loss
EXTRA_ARGS="--local_files_only --epochs 8 --batch_size 8 --eval_batch_size 16 --experiment_name exp2_weak_pair --pair_loss_weight 0.03 --pair_margin 0.20"

# Exp3: only light happy CE up-weighting
EXTRA_ARGS="--local_files_only --epochs 8 --batch_size 8 --eval_batch_size 16 --experiment_name exp3_happy_ce --happy_ce_weight 1.10"
```

Watch progress:

```bash
tail -f outputs/hdsa_erc_iemocap/epoch_results.txt
```

## Outputs

Training writes:

- `best_dev_model.pt`
- `best_test_model.pt`
- `last_model.pt` only when `--save_last` is enabled
- `epoch_results.txt`
- `epoch_metrics.csv`
- `metrics.jsonl`
- `final_metrics.json`
- `experiment_result.json`
- `dev_classification_report.json`
- `test_classification_report.json`
- `dev_predictions.csv`
- `test_predictions.csv`

Every epoch logs `loss_total`, `loss_ce`, `loss_proto`, `loss_compact`, optional ablation losses, dev/test metrics, per-class F1, target confusion-pair counts, anchor similarity stats, OT entropy/max-prob stats, OT assignment counts, hard assignment counts, selected top-ratio EMA counts, class-wise OT max-prob quantiles, EMA update counts, and current best dev/test summaries. By default, top-ratio EMA, fallback EMA, pairwise losses, intensity loss, and happy CE reweighting are off so Exp0 restores the anti-collapse baseline.
