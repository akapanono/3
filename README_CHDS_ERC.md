# CHDS-ERC

This repository now contains an implementation of the model described in
`CHDS_ERC_single_model_design.md`: Cluster-guided Hyperspherical Domain
Sub-anchor Network for Emotion Recognition in Conversation.

## Train

```bash
python -m chds_erc.train ^
  --dataset_dir data/IEMOCAP ^
  --model_name_or_path E:\AI_learn_model\ERC\EACL\pretrained\sup-simcse-roberta-large ^
  --local_files_only ^
  --output_dir outputs/chds_erc_iemocap
```

If `--model_name_or_path` is left as `princeton-nlp/sup-simcse-roberta-large`,
the training script will automatically use the local Sup-SimCSE directory above
when it exists.

The loader also supports:

```bash
python -m chds_erc.train --dataset_dir data/MELD --output_dir outputs/chds_erc_meld
python -m chds_erc.train --dataset_dir data/EmoryNLP --output_dir outputs/chds_erc_emory
```

## Server Background Training

On a Linux server, run training in the background with:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATASET_DIR=data/IEMOCAP \
MODEL_NAME_OR_PATH=princeton-nlp/sup-simcse-roberta-large \
OUTPUT_DIR=outputs/chds_erc_iemocap \
EXTRA_ARGS="--epochs 8 --batch_size 8 --eval_batch_size 16" \
bash scripts/run_chds_erc_background.sh
```

Then monitor live epoch results:

```bash
tail -f outputs/chds_erc_iemocap/epoch_results.txt
```

The background script also writes `stdout.log` and `train.pid` in the output
directory.

Useful CPU smoke test:

```bash
python -m chds_erc.train ^
  --dataset_dir data/IEMOCAP ^
  --model_name_or_path bert-base-uncased ^
  --local_files_only ^
  --freeze_encoder ^
  --anchor_dim 64 ^
  --epochs 1 ^
  --batch_size 2 ^
  --eval_batch_size 2 ^
  --max_train_samples 8 ^
  --max_dev_samples 4 ^
  --max_test_samples 4 ^
  --output_dir outputs/smoke
```

## Outputs

Training writes:

- `best_model.pt` and `last_model.pt`
- `metadata.json`
- `metrics.jsonl`
- `epoch_metrics.csv`
- `epoch_results.txt`
- `final_metrics.json`
- `dev_classification_report.json`
- `test_classification_report.json`
- `dev_predictions.csv`
- `test_predictions.csv`
- `initial_anchor_report.json`
- `final_anchor_report.json`

Each epoch records accuracy, macro/weighted precision, macro/weighted recall,
macro/weighted F1, CE loss, all auxiliary losses, and anchor similarity stats.
The anchor reports include same-class / different-class cosine similarity
statistics and per-class sub-anchor assignment counts.

## Predict

```bash
python -m chds_erc.predict ^
  --checkpoint outputs/chds_erc_iemocap/best_model.pt ^
  --local_files_only ^
  --speaker A ^
  --text "I cannot believe you did that." ^
  --history "B::I already apologized."
```
