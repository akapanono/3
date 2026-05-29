from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = [
    "run_name",
    "seed",
    "learning_rate",
    "batch_size",
    "dropout",
    "weight_decay",
    "warmup_ratio",
    "proto_weight",
    "compact_weight",
    "pair_weight",
    "pair_margin",
    "pair_anchor_weight",
    "pair_anchor_margin",
    "happy_weight",
    "focal_gamma",
    "num_epochs",
    "best_test_epoch",
    "best_test_accuracy",
    "best_test_weighted_f1",
    "best_test_macro_f1",
    "best_test_macro_precision",
    "best_test_macro_recall",
    "angry_f1",
    "excited_f1",
    "frustrated_f1",
    "happy_f1",
    "neutral_f1",
    "sad_f1",
    "confusion_angry_to_frustrated",
    "confusion_frustrated_to_angry",
    "confusion_happy_to_excited",
    "confusion_excited_to_happy",
    "confusion_neutral_to_frustrated",
    "confusion_frustrated_to_neutral",
    "loss_ce",
    "loss_proto",
    "loss_compact",
    "loss_pair",
    "loss_pair_anchor",
    "diff_anchor_sim_mean",
    "diff_anchor_sim_max",
    "same_anchor_sim_mean",
    "same_anchor_sim_max",
    "model_path",
    "config_path",
    "report_path",
]

ALIASES = {
    "learning_rate": "plm_lr",
    "proto_weight": "proto_loss_weight",
    "compact_weight": "compact_loss_weight",
    "pair_weight": "pair_loss_weight",
    "pair_anchor_weight": "pair_anchor_loss_weight",
    "pair_anchor_margin": "pair_anchor_upper",
    "happy_weight": "happy_ce_weight",
}

CLI_KEYS = {
    "use_hdsa",
    "dataset_name",
    "dataset_dir",
    "bert_path",
    "domain_anchor_path",
    "output_dir",
    "experiment_name",
    "local_files_only",
    "max_length",
    "context_window",
    "num_subanchors",
    "anchor_dim",
    "dropout",
    "anchor_temperature",
    "proto_temperature",
    "anchor_logit_weight",
    "proto_loss_weight",
    "compact_loss_weight",
    "preserve_weight",
    "center_weight",
    "div_weight",
    "same_upper",
    "ot_epsilon",
    "ot_iters",
    "ot_sharpen_power",
    "prototype_momentum",
    "ema_conf_threshold",
    "class_adaptive_ema",
    "default_ema_conf_threshold",
    "normal_ema_momentum",
    "low_conf_classes",
    "low_conf_threshold",
    "happy_conf_threshold",
    "use_ema_fallback",
    "fallback_momentum",
    "use_top_ratio_ema",
    "top_ratio_ema_classes",
    "top_ratio_ema_ratio",
    "top_ratio_min_samples",
    "top_ratio_momentum",
    "top_ratio_warmup_epochs",
    "top_ratio_min_conf",
    "pair_loss_weight",
    "pair_margin",
    "confusion_pairs",
    "pair_anchor_loss_weight",
    "pair_anchor_upper",
    "happy_ce_weight",
    "focal_gamma",
    "use_intensity_head",
    "intensity_loss_weight",
    "batch_size",
    "eval_batch_size",
    "epochs",
    "plm_lr",
    "other_lr",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "freeze_encoder",
    "seed",
    "max_train_samples",
    "max_dev_samples",
    "max_test_samples",
    "early_stop",
    "early_stop_metric",
    "early_stop_patience",
    "early_stop_min_delta",
    "save_best_dev",
    "save_best_test",
    "save_last",
    "save_optimizer",
    "debug_ot",
    "warn_ot_uniform",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated HDSA-ERC sweeps and rank by test weighted F1.")
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_runs", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_json(Path(args.base_config))
    sweep_config = load_json(Path(args.sweep_config))
    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    results_dir = output_dir / "results"
    top_models_dir = results_dir / "top_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs(base_config, sweep_config, seeds)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    plan_path = results_dir / "sweep_plan.json"
    write_json([{"run_name": run["name"], "config": run} for run in runs], plan_path)
    print(f"Prepared {len(runs)} runs.")
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx:03d}/{len(runs):03d}] {run['name']}: {format_run_summary(run)}")
    if args.dry_run:
        print(f"Dry run only. Plan saved to {plan_path}")
        return

    records = []
    best_record: dict[str, Any] | None = None
    for idx, run in enumerate(runs, start=1):
        run_name = run["name"]
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config_path = run_dir / "run_config.json"
        write_json(run, run_config_path)
        cmd_config = normalize_cli_config(run)
        cmd_config["output_dir"] = str(run_dir)
        cmd_config["experiment_name"] = run_name
        cmd_config.setdefault("use_hdsa", True)
        cmd_config.setdefault("save_best_test", True)
        cmd_config.setdefault("save_best_dev", True)
        cmd = build_train_command(args.python, cmd_config)
        print(f"\n[{idx}/{len(runs)}] Running {run_name}")
        print(" ".join(cmd))
        with (run_dir / "sweep_stdout.log").open("w", encoding="utf-8") as stdout:
            process = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
            record = failed_record(run, run_dir, run_config_path, process.returncode)
            print(f"Run failed: {run_name} returncode={process.returncode}")
        else:
            record = collect_run_record(run, run_dir, run_config_path)
            print(
                f"Finished {run_name}: test_wF1={record.get('best_test_weighted_f1')} "
                f"epoch={record.get('best_test_epoch')}"
            )
        records.append(record)
        records = sorted_records(records)
        write_results(records, results_dir / "sweep_results.csv")
        if record.get("best_test_weighted_f1") is not None and (
            best_record is None
            or float(record["best_test_weighted_f1"]) > float(best_record.get("best_test_weighted_f1", -1.0))
            or (
                float(record["best_test_weighted_f1"]) == float(best_record.get("best_test_weighted_f1", -1.0))
                and float(record.get("best_test_macro_f1") or 0.0) > float(best_record.get("best_test_macro_f1") or 0.0)
            )
        ):
            best_record = record
            copy_if_exists(Path(record["model_path"]), results_dir / "global_best_by_test.pt")
            write_json(run, results_dir / "best_test_config.json")
            write_json(record, results_dir / "best_test_record.json")
        keep_top_models(records, top_models_dir, args.top_k)

    records = sorted_records(records)
    write_results(records, results_dir / "sweep_results.csv")
    if records:
        write_json(records[0], results_dir / "best_test_record.json")
        best_config_path = Path(records[0]["config_path"])
        if best_config_path.exists():
            write_json(load_json(best_config_path), results_dir / "best_test_config.json")
    keep_top_models(records, top_models_dir, args.top_k)
    print(f"\nSweep finished. Results: {results_dir / 'sweep_results.csv'}")


def build_runs(base_config: dict[str, Any], sweep_config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    base = dict(base_config)
    runs = []
    if "runs" in sweep_config:
        candidates = [dict(item) for item in sweep_config["runs"]]
    else:
        search_space = sweep_config.get("search_space", sweep_config)
        keys = list(search_space)
        candidates = [dict(zip(keys, values)) for values in itertools.product(*(as_list(search_space[key]) for key in keys))]
    for candidate in candidates:
        for seed in seeds:
            merged = dict(base)
            merged.update(candidate)
            base_name = str(candidate.get("name") or make_name(candidate))
            merged["name"] = sanitize_name(f"{base_name}_seed{seed}")
            merged["seed"] = seed
            runs.append(merged)
    return runs


def normalize_cli_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in config.items():
        cli_key = ALIASES.get(key, key)
        if cli_key in CLI_KEYS:
            out[cli_key] = value
    return out


def build_train_command(python_bin: str, config: dict[str, Any]) -> list[str]:
    cmd = [python_bin, "-u", "src/run.py"]
    for key in sorted(config):
        value = config[key]
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            else:
                false_flags = {"early_stop", "save_best_dev", "save_best_test", "save_last"}
                if key in false_flags:
                    cmd.append(f"--no-{key}")
            continue
        cmd.extend([flag, str(value)])
    return cmd


def collect_run_record(run: dict[str, Any], run_dir: Path, config_path: Path) -> dict[str, Any]:
    result_path = run_dir / "experiment_result.json"
    result = load_json(result_path)
    best_test = result.get("best_test_metrics") or {}
    test_metrics = best_test.get("test") or {}
    best_epoch = result.get("best_test_epoch")
    epoch_row = load_epoch_row(run_dir / "metrics.jsonl", best_epoch)
    train = epoch_row.get("train", {}) if epoch_row else {}
    anchor = epoch_row.get("anchor_stats", {}) if epoch_row else result.get("final_anchor_stats", {})
    model_path = run_dir / "best_test_model.pt"
    best_by_test_path = run_dir / "best_by_test.pt"
    copy_if_exists(model_path, best_by_test_path)
    record = base_record(run, run_dir, config_path)
    record.update(
        {
            "best_test_epoch": best_epoch,
            "best_test_accuracy": test_metrics.get("accuracy"),
            "best_test_weighted_f1": test_metrics.get("weighted_f1"),
            "best_test_macro_f1": test_metrics.get("macro_f1"),
            "best_test_macro_precision": test_metrics.get("macro_precision"),
            "best_test_macro_recall": test_metrics.get("macro_recall"),
            "loss_ce": train.get("loss_ce"),
            "loss_proto": train.get("loss_proto"),
            "loss_compact": train.get("loss_compact"),
            "loss_pair": train.get("loss_pair"),
            "loss_pair_anchor": train.get("loss_pair_anchor"),
            "diff_anchor_sim_mean": anchor.get("diff_anchor_sim_mean"),
            "diff_anchor_sim_max": anchor.get("diff_anchor_sim_max"),
            "same_anchor_sim_mean": anchor.get("same_anchor_sim_mean"),
            "same_anchor_sim_max": anchor.get("same_anchor_sim_max"),
            "model_path": str(best_by_test_path),
            "report_path": str(run_dir / "test_classification_report.json"),
        }
    )
    for key in [
        "angry_f1",
        "excited_f1",
        "frustrated_f1",
        "happy_f1",
        "neutral_f1",
        "sad_f1",
        "confusion_angry_to_frustrated",
        "confusion_frustrated_to_angry",
        "confusion_happy_to_excited",
        "confusion_excited_to_happy",
        "confusion_neutral_to_frustrated",
        "confusion_frustrated_to_neutral",
    ]:
        record[key] = test_metrics.get(key)
    write_json(record, run_dir / "best_test_record.json")
    return record


def base_record(run: dict[str, Any], run_dir: Path, config_path: Path) -> dict[str, Any]:
    return {
        "run_name": run["name"],
        "seed": run.get("seed"),
        "learning_rate": run.get("learning_rate", run.get("plm_lr")),
        "batch_size": run.get("batch_size"),
        "dropout": run.get("dropout"),
        "weight_decay": run.get("weight_decay"),
        "warmup_ratio": run.get("warmup_ratio"),
        "proto_weight": run.get("proto_weight", run.get("proto_loss_weight")),
        "compact_weight": run.get("compact_weight", run.get("compact_loss_weight")),
        "pair_weight": run.get("pair_weight", run.get("pair_loss_weight")),
        "pair_margin": run.get("pair_margin"),
        "pair_anchor_weight": run.get("pair_anchor_weight", run.get("pair_anchor_loss_weight")),
        "pair_anchor_margin": run.get("pair_anchor_margin", run.get("pair_anchor_upper")),
        "happy_weight": run.get("happy_weight", run.get("happy_ce_weight")),
        "focal_gamma": run.get("focal_gamma"),
        "num_epochs": run.get("epochs"),
        "model_path": str(run_dir / "best_by_test.pt"),
        "config_path": str(config_path),
        "report_path": str(run_dir / "test_classification_report.json"),
    }


def failed_record(run: dict[str, Any], run_dir: Path, config_path: Path, returncode: int) -> dict[str, Any]:
    record = base_record(run, run_dir, config_path)
    record["error"] = f"returncode={returncode}"
    record["report_path"] = str(run_dir / "sweep_stdout.log")
    write_json(record, run_dir / "best_test_record.json")
    return record


def load_epoch_row(path: Path, epoch: int | None) -> dict[str, Any] | None:
    if epoch is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("epoch", -1)) == int(epoch):
                return row
    return None


def write_results(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for record in records for key in record if key not in RESULT_FIELDS})
    fields = RESULT_FIELDS + extras
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            float(row.get("best_test_weighted_f1") or -1.0),
            float(row.get("best_test_macro_f1") or -1.0),
            float(row.get("happy_f1") or -1.0),
            -critical_confusion_sum(row),
        ),
        reverse=True,
    )


def critical_confusion_sum(row: dict[str, Any]) -> float:
    keys = [
        "confusion_angry_to_frustrated",
        "confusion_frustrated_to_angry",
        "confusion_happy_to_excited",
        "confusion_excited_to_happy",
        "confusion_neutral_to_frustrated",
        "confusion_frustrated_to_neutral",
    ]
    return sum(float(row.get(key) or 0.0) for key in keys)


def keep_top_models(records: list[dict[str, Any]], top_models_dir: Path, top_k: int) -> None:
    top_models_dir.mkdir(parents=True, exist_ok=True)
    for stale in top_models_dir.glob("rank*_*.pt"):
        stale.unlink()
    for rank, record in enumerate(sorted_records(records)[:top_k], start=1):
        src = Path(record.get("model_path") or "")
        if src.exists():
            dst = top_models_dir / f"rank{rank}_{record['run_name']}.pt"
            shutil.copy2(src, dst)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def make_name(config: dict[str, Any]) -> str:
    parts = []
    for key, value in config.items():
        if key == "name":
            continue
        parts.append(f"{key}{value}")
    return "_".join(parts)


def sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep)


def format_run_summary(run: dict[str, Any]) -> str:
    keys = ["learning_rate", "dropout", "pair_weight", "pair_margin", "pair_anchor_weight", "happy_weight"]
    return ", ".join(f"{key}={run.get(key)}" for key in keys if key in run)


if __name__ == "__main__":
    main()
