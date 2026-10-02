#!/usr/bin/env python3
"""Preflight validation for paper-aligned LabGrasp-Deploy-Real training."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml


EXPECTED_SUPERVISION = {
    "Affordance_Grounding": {"mask": True, "pose": False, "language": True},
    "Pose_Regression": {"mask": False, "pose": True, "language": True},
    "VQA": {"mask": False, "pose": False, "language": True},
}


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_full.yaml")
    ap.add_argument("--max-errors", type=int, default=30)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(cfg["data_root"])
    excluded = {str(x).lower().replace("_", " ")
                for x in cfg.get("exclude_classes", [])}
    errors = []
    split_counts = {}
    split_images = {}
    all_record_ids = set()

    def error(message: str):
        if len(errors) < args.max_errors:
            errors.append(message)

    for split, key in (("train", "train_jsonl"),
                       ("val", "val_jsonl"),
                       ("test", "test_jsonl")):
        path = Path(cfg[key])
        if not path.is_file():
            error(f"{split}: missing JSONL: {path}")
            continue
        counts = Counter()
        kept_counts = Counter()
        images = set()
        image_basenames = set()
        ag_pairs = set()
        pr_pairs = []

        for line_no, item in load_jsonl(path):
            where = f"{path}:{line_no}"
            record_id = item.get("id")
            if not record_id:
                error(f"{where}: missing record id")
            elif record_id in all_record_ids:
                error(f"{where}: duplicate record id={record_id!r}")
            else:
                all_record_ids.add(record_id)
            dtype = item.get("data_type")
            counts[dtype] += 1
            if dtype not in EXPECTED_SUPERVISION:
                error(f"{where}: unknown data_type={dtype!r}")
                continue
            if item.get("supervision") != EXPECTED_SUPERVISION[dtype]:
                error(
                    f"{where}: supervision={item.get('supervision')} expected="
                    f"{EXPECTED_SUPERVISION[dtype]}"
                )
            if item.get("split") != split:
                error(f"{where}: record split={item.get('split')!r}, expected {split!r}")

            image_rel = item.get("image", "")
            image_path = root / image_rel
            images.add(image_rel)
            image_basenames.add(Path(image_rel).name)
            if not image_path.is_file():
                error(f"{where}: missing image {image_path}")

            gt = item.get("ground_truth") or {}
            obj = str(gt.get("object_class", "")).lower().replace("_", " ")
            is_excluded = obj in excluded
            if is_excluded:
                counts["excluded_by_config"] += 1
            else:
                kept_counts[dtype] += 1
            pair = item.get("pair_id") or item.get("related_pair_id")
            if dtype == "Affordance_Grounding":
                mask_rel = gt.get("mask_path")
                if not mask_rel or not (root / mask_rel).is_file():
                    error(f"{where}: missing AG mask {mask_rel!r}")
                if gt.get("pose") is not None:
                    error(f"{where}: AG must not carry trainable pose")
                if not pair:
                    error(f"{where}: AG record is missing pair_id")
                elif not is_excluded:
                    ag_pairs.add(pair)
            elif dtype == "Pose_Regression":
                pose = gt.get("pose")
                if not isinstance(pose, dict) or not all(
                    key in pose for key in ("x", "y", "angle", "width")
                ):
                    error(f"{where}: invalid PR pose {pose!r}")
                elif not all(
                    isinstance(pose[key], (int, float))
                    and math.isfinite(float(pose[key]))
                    for key in ("x", "y", "angle", "width")
                ) or float(pose["width"]) <= 0:
                    error(f"{where}: non-finite or non-positive PR pose {pose!r}")
                if gt.get("mask_path") is not None:
                    error(f"{where}: PR must not carry trainable mask_path")
                if not pair:
                    error(f"{where}: PR record is missing pair_id")
                elif not is_excluded:
                    pr_pairs.append((pair, where))
            elif dtype == "VQA":
                if gt.get("mask_path") is not None or gt.get("pose") is not None:
                    error(f"{where}: VQA must be language-only")

            conversations = item.get("conversations") or []
            if not conversations or conversations[-1].get("from") != "assistant":
                error(f"{where}: missing final assistant target")
            else:
                target = str(conversations[-1].get("value", ""))
                if dtype == "Affordance_Grounding" and "<GRASP_MASK>" not in target:
                    error(f"{where}: AG target is missing <GRASP_MASK>")
                if dtype == "Pose_Regression" and "<GRASP_POSE>" not in target:
                    error(f"{where}: PR target is missing <GRASP_POSE>")
                if dtype == "VQA" and (
                    "<GRASP_MASK>" in target or "<GRASP_POSE>" in target
                ):
                    error(f"{where}: VQA target contains a structured-output token")
        for pair, where in pr_pairs:
            if pair not in ag_pairs:
                error(f"{where}: PR pair has no valid AG in the same split: {pair}")
        for dtype in EXPECTED_SUPERVISION:
            if kept_counts[dtype] == 0:
                error(f"{split}: no usable {dtype} records after class filtering")
        counts["kept_total"] = sum(kept_counts.values())
        split_counts[split] = counts
        split_images[split] = image_basenames

    # The provided dataset is source-image-disjoint; detect accidental overlap.
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_images.get(left, set()) & split_images.get(right, set())
        if overlap:
            error(f"image leakage {left}/{right}: {len(overlap)} shared basenames")

    print(f"Config: {config_path}")
    print(f"Dataset root: {root}")
    print(f"Excluded classes: {sorted(excluded) or 'none'}")
    for split in ("train", "val", "test"):
        print(f"{split:5s}: {dict(split_counts.get(split, {}))}")

    if errors:
        print(f"\nFAILED: {len(errors)} reported problem(s)")
        for message in errors:
            print(" -", message)
        return 1
    print("\nPASSED: dataset paths, heterogeneous supervision, and AG-PR pairs are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
