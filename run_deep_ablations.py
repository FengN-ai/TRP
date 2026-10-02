#!/usr/bin/env python3
"""Prepare, train and evaluate independent VA-Fusion/CAPD submechanism ablations.

Examples (run from the repository root):
  python run_deep_ablations.py prepare --root deep_ablation_runs
  CUDA_VISIBLE_DEVICES=0,1 python run_deep_ablations.py run --root deep_ablation_runs --nproc 2
  python run_deep_ablations.py run --root deep_ablation_runs --variants capd_no_progressive --seeds 42 43 44
  python run_deep_ablations.py evaluate --root deep_ablation_runs
  python run_deep_ablations.py summarize --root deep_ablation_runs

All variants are independently trained from the configured pretrained backbones.
Existing full-model checkpoints are never used to initialize an ablated run.
Training/loss/metric/optimizer/scheduler implementations are imported unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
# This exact prior runner differs only in evaluation checkpoint loading. Its
# prepared/trained suites remain valid if ALL model/training/data/loss source
# hashes match. Do not accept arbitrary runner revisions or rewrite history.
_PRE_QUANT_LOADER_FIX_SHA256 = "8b1829fe923af030a83619b6ea3863a0a2a83c94a4c944b57a7fc486c2567a58"
DEFAULTS = {
    "use_capd": True, "use_vaf": True, "use_shared_decoder": True,
    "vaf_fixed_gate": None,
    "capd_use_confidence_weighting": True,
    "capd_use_progressive_sampling": True,
    "capd_fixed_alpha": 0.9,
    "capd_window_size": 5,
}
VARIANTS = {
    "full": ("Full model reference", {}),
    "vaf_fixed_gate": ("Fixed DINO/SigLIP gate = 0.5; keep gate supervision", {"vaf_fixed_gate": 0.5}),
    "vaf_no_gate_supervision": ("Learned gate without viewpoint auxiliary supervision", {"lambda_vp": 0.0}),
    "capd_uniform_weights": ("Uniform mean inside the same 5x5 angle-sampling window", {"capd_use_confidence_weighting": False}),
    "capd_no_progressive": ("Constant training sampling alpha = 0.9; keep consistency-loss schedule", {"capd_use_progressive_sampling": False}),
    "capd_point_sampling": ("Single-point bilinear angle sampling (1x1 window)", {"capd_window_size": 1}),
    "capd_no_consistency": ("Remove centroid consistency loss only", {"lambda_consist": 0.0}),
}
METRICS = ["ciou", "ciou_fixed", "miou", "giou", "soft_iou",
           "translation_error_px", "angle_error_deg", "width_error_px",
           "loss_total", "loss_ce", "loss_mask", "loss_pose"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    names = ("train.py", "model.py", "dataset.py", "utils.py",
             "run_deep_ablations.py")
    return {name: file_hash(SCRIPT_DIR / name) for name in names}


def compatible_sources(recorded):
    current = sources()
    if recorded == current:
        return True
    previous = dict(current)
    previous["run_deep_ablations.py"] = _PRE_QUANT_LOADER_FIX_SHA256
    return recorded == previous


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    from train import VLAConfig
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(values) - set(VLAConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = VLAConfig()
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg, values


def prepare(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg, base = load_config(args.base_config)
    # A full reference is essential: do not layer these experiments on a
    # previously disabled module, loss or nonstandard sampling mechanism.
    for key, expected in DEFAULTS.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"Base config must use full-model {key}={expected!r}")
    if cfg.lambda_consist <= 0 or cfg.lambda_vp <= 0:
        raise ValueError("Full reference requires lambda_consist > 0 and lambda_vp > 0")
    if cfg.capd_alpha_end != 0.9:
        raise ValueError("This suite compares progressive -> 0.9 against fixed 0.9; use capd_alpha_end: 0.9")
    if cfg.stage1_epochs <= 0 or cfg.stage2_epochs <= 0:
        raise ValueError("This suite requires both original training stages")
    if len(args.seeds) != len(set(args.seeds)) or len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate seeds or variants are not allowed")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must lie in [0, 2**32)")
    source_hashes = sources()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not compatible_sources(manifest["source_sha256"]) or manifest["base_config"] != base:
            raise ValueError("Source/base config differs from the existing suite. Use a new --root to avoid mixing experiments.")
    else:
        manifest = {"created_utc": utc_now(), "base_config_path": str(Path(args.base_config).resolve()),
                    "base_config": base, "source_sha256": source_hashes, "runs": []}
    indexed = {run["id"]: run for run in manifest["runs"]}
    for variant in args.variants:
        description, changes = VARIANTS[variant]
        for seed in args.seeds:
            run_id = f"{variant}/seed_{seed}"
            folder = root / variant / f"seed_{seed}"
            values = copy.deepcopy(base)
            for key, value in DEFAULTS.items():
                values.setdefault(key, value)
            values.update(changes)
            values.update(seed=seed, resume_from="", output_dir=str(folder / "checkpoints"),
                          log_dir=str(folder / "logs"))
            config_path = folder / "config.yaml"
            if config_path.exists():
                existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if existing != values:
                    raise ValueError(f"Refusing to replace a differing experiment config: {config_path}")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
            indexed[run_id] = {"id": run_id, "variant": variant, "seed": seed,
                               "description": description, "overrides": changes,
                               "config": str(config_path), "folder": str(folder),
                               "config_sha256": file_hash(config_path)}
    manifest["runs"] = list(indexed.values())
    write_json(manifest_path, manifest)
    print(f"Prepared {len(args.variants) * len(args.seeds)} runs in {root}", flush=True)
    return manifest


def checked_manifest(root):
    manifest = read_json(Path(root) / "manifest.json")
    if not compatible_sources(manifest["source_sha256"]):
        raise ValueError("Source code differs from the suite manifest; do not mix implementations")
    for run in manifest["runs"]:
        if file_hash(run["config"]) != run["config_sha256"]:
            raise ValueError(f"Config changed after preparation: {run['id']}")
    return manifest


def run_command(command, logfile, env=None):
    print(shlex.join([str(part) for part in command]), flush=True)
    print(f"  Log: {logfile}", flush=True)
    # Inherit terminal output via a streaming tee so long jobs remain visible.
    with Path(logfile).open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=SCRIPT_DIR, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = child.wait()
        except BaseException:
            # Terminate the launcher and its workers when the suite is stopped.
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        finally:
            child.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def evaluate_run(run, checkpoint_name="best_model.pth"):
    folder = Path(run["folder"])
    checkpoint = folder / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}. No silent best-to-last fallback; use --checkpoint last_model.pth explicitly if necessary.")
    output = folder / ("test_metrics.json" if checkpoint_name == "best_model.pth" else "test_metrics_last.json")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "eval-one",
               "--config", run["config"], "--checkpoint-path", str(checkpoint), "--output", str(output)]
    run_command(command, folder / "test_console.log")


def run_suite(args):
    manifest = prepare(args)
    # World size changes effective batch size and the optimizer-step schedule.
    # Keep it identical across all variants/seeds sharing a comparison root.
    recorded_nproc = manifest.get("nproc_per_node")
    if recorded_nproc is not None and recorded_nproc != args.nproc:
        raise ValueError(f"This suite uses --nproc {recorded_nproc}; use a new --root to change GPU count")
    manifest["nproc_per_node"] = args.nproc
    write_json(Path(args.root).resolve() / "manifest.json", manifest)
    selected = [run for run in manifest["runs"]
                if run["variant"] in args.variants and run["seed"] in args.seeds]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
    # Always use torchrun, even for a single GPU: the original trainer expects
    # RANK/WORLD_SIZE for its single-GPU/DDP setup semantics.
    for run in selected:
        folder = Path(run["folder"])
        status_path = folder / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        if status.get("training_completed"):
            print(f"Skip completed training: {run['id']}", flush=True)
        else:
            checkpoints = folder / "checkpoints"
            occupied = checkpoints.exists() and any(checkpoints.iterdir())
            if occupied and not args.resume:
                raise ValueError(f"{run['id']} has incomplete checkpoints. Use --resume for this same run or a new --root.")
            command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={args.nproc}", str(SCRIPT_DIR / "train.py"), "--config", run["config"]]
            if occupied:
                last = checkpoints / "last_model.pth"
                if not last.is_file():
                    raise FileNotFoundError(f"Cannot resume without {last}")
                command += ["--resume", str(last)]
            attempt = {"started_utc": utc_now(), "command": command,
                       "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "nproc": args.nproc}
            start = time.perf_counter()
            try:
                run_command([sys.executable, str(SCRIPT_DIR / "check_labgrasp_training.py"),
                             "--config", run["config"]], folder / "preflight.log", env)
                run_command(command, folder / "train_console.log", env)
                attempt["success"] = True
                status["training_completed"] = True
            except BaseException as exc:
                attempt.update(success=False, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                attempt["wall_time_s"] = time.perf_counter() - start
                status.setdefault("attempts", []).append(attempt)
                write_json(status_path, status)
        if not args.no_evaluate:
            evaluate_run(run, args.checkpoint)
            status["last_test_evaluation_utc"] = utc_now()
            write_json(status_path, status)
        summarize(args.root, args.checkpoint)


def eval_one(args):
    import torch
    from torch.utils.data import DataLoader
    from dataset import VLADataset
    from train import evaluate, set_global_seed
    from checkpoint_compat import load_evaluation_checkpoint
    from model import build_vla_model

    if not torch.cuda.is_available():
        raise RuntimeError("Test evaluation requires the CUDA server and configured pretrained weights/data")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("eval-one is a single-GPU subprocess; do not launch it with torchrun")
    cfg, values = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_global_seed(cfg.seed)
    model = build_vla_model(cfg, device=device)
    for parameter in model.parameters():
        if parameter.device.type == "cpu":
            parameter.data = parameter.data.to(device)
    for buffer in model.buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to(device)
    # Restore 4-bit quantization state while preserving strict key checks.
    checkpoint, _, restoration = load_evaluation_checkpoint(args.checkpoint_path, model, strict=True)
    stage, step = int(checkpoint.get("stage", 2)), int(checkpoint.get("global_step", 0))
    log_vars = {key: checkpoint.get("log_vars", {}).get(
        key, getattr(model, "log_var_" + key)).detach().to(device)
        for key in ("text", "mask", "pose")}
    selection = {"stage": stage, "global_step": step, "best_validation_ciou": checkpoint.get("best_miou")}
    del checkpoint
    dataset = VLADataset(jsonl_paths=[cfg.test_jsonl], data_root=cfg.data_root,
                         tokenizer=model.tokenizer, max_length=cfg.max_seq_len,
                         dino_size=cfg.dino_size, siglip_size=cfg.siglip_size, mask_size=cfg.img_size,
                         augment=False, exclude_classes=cfg.exclude_classes, strict_data=True)
    loader = DataLoader(dataset, batch_size=cfg.per_device_batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    # Existing evaluate() contains the project's exact mask/pose metrics. Test
    # records are never used for checkpoint selection; evaluate every batch.
    with torch.no_grad():
        metrics = evaluate(model, loader, device, cfg, stage, step, log_vars,
                           split_name="Test", max_batches=None)
    metrics = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
               for key, value in metrics.items()}
    write_json(args.output, {"evaluated_utc": utc_now(), "config_sha256": file_hash(args.config),
                             "config": values, "source_sha256": sources(),
                             "checkpoint": str(Path(args.checkpoint_path).resolve()),
                             "checkpoint_selection": selection, "test_samples": len(dataset),
                             "quantization_restoration": restoration,
                             "evaluation_loader_sha256": file_hash(SCRIPT_DIR / "checkpoint_compat.py"),
                             "metrics": metrics})
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def summarize(root, checkpoint_name="best_model.pth"):
    root = Path(root).resolve()
    manifest = checked_manifest(root)
    per_seed, groups = [], {}
    filename = "test_metrics.json" if checkpoint_name == "best_model.pth" else "test_metrics_last.json"
    for run in manifest["runs"]:
        row = {"variant": run["variant"], "seed": run["seed"], "status": "not_evaluated",
               "checkpoint": checkpoint_name}
        path = Path(run["folder"]) / filename
        if path.exists():
            result = read_json(path)
            if result["config_sha256"] != run["config_sha256"] or not compatible_sources(result["source_sha256"]):
                raise ValueError(f"Stale evaluation result: {path}")
            row.update(status="evaluated", **{key: result["metrics"].get(key) for key in METRICS})
            groups.setdefault(run["variant"], []).append(row)
        per_seed.append(row)
    suffix = "" if checkpoint_name == "best_model.pth" else "_last"
    with (root / f"results_per_seed{suffix}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "seed", "status", "checkpoint"] + METRICS)
        writer.writeheader()
        writer.writerows(per_seed)
    summary = []
    for variant in dict.fromkeys(run["variant"] for run in manifest["runs"]):
        rows = groups.get(variant, [])
        record = {"variant": variant,
                  "expected_seeds": sum(run["variant"] == variant for run in manifest["runs"]),
                  "evaluated_seeds": len(rows)}
        for key in METRICS:
            values = [row[key] for row in rows if row[key] is not None]
            record[key + "_n"] = len(values)
            record[key + "_mean"] = statistics.mean(values) if values else None
            # One seed cannot estimate seed-to-seed variability.
            record[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(record)
    with (root / f"results_summary{suffix}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["variant", "expected_seeds", "evaluated_seeds"]
        fields += [key + ending for key in METRICS for ending in ("_n", "_mean", "_std")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary: {root / ('results_summary' + suffix + '.csv')}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "run"):
        p = commands.add_parser(name)
        p.add_argument("--root", default="deep_ablation_runs")
        p.add_argument("--base-config", default=str(SCRIPT_DIR / "config_full.yaml"))
        p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
        p.add_argument("--seeds", type=int, nargs="+", default=[42])
        if name == "run":
            p.add_argument("--nproc", type=int, default=1)
            p.add_argument("--resume", action="store_true", help="Resume each unfinished variant only from its own last_model.pth")
            p.add_argument("--no-evaluate", action="store_true")
            p.add_argument("--checkpoint", choices=["best_model.pth", "last_model.pth"], default="best_model.pth")
    for name in ("evaluate", "summarize"):
        p = commands.add_parser(name)
        p.add_argument("--root", default="deep_ablation_runs")
        p.add_argument("--checkpoint", choices=["best_model.pth", "last_model.pth"], default="best_model.pth")
    p = commands.add_parser("eval-one", help="Evaluate one trained run using its generated config")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode == "run" and args.nproc < 1:
        parser.error("--nproc must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "prepare":
        prepare(args)
        summarize(args.root)
    elif args.mode == "run":
        run_suite(args)
    elif args.mode == "eval-one":
        eval_one(args)
    elif args.mode == "evaluate":
        for run in checked_manifest(args.root)["runs"]:
            evaluate_run(run, args.checkpoint)
            summarize(args.root, args.checkpoint)
    else:
        summarize(args.root, args.checkpoint)


if __name__ == "__main__":
    main()
