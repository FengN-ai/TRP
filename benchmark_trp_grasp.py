#!/usr/bin/env python3
r"""Standalone TRP-Grasp cost benchmark. Does not edit the training/model code.

Keep checkpoint_compat.py beside this script for strict 4-bit checkpoint loading.

Additional resource-monitor dependencies:
    pip install psutil nvidia-ml-py

Inference, both spatial branches, fixed text length, one image per request:
    python benchmark_trp_grasp.py infer --config config_full.yaml \
        --checkpoint /path/to/best_model.pth --seq-len 128 \
        --task both --batch-size 1 --output benchmark_full

Replay one real batch with the unmodified validation dataset preprocessing:
    python benchmark_trp_grasp.py infer --config config_full.yaml \
        --checkpoint /path/to/best_model.pth --input data --output benchmark_data

Measure an actual training job, including DDP workers and validation/checkpoints:
    python benchmark_trp_grasp.py monitor --gpu-ids 0,1 --output training_cost -- \
        torchrun --nproc_per_node=2 train.py --config config_full.yaml

GPU IDs for monitor are PHYSICAL NVML indices/UUIDs, not CUDA-visible indices.
The training command keeps the caller's CUDA_VISIBLE_DEVICES environment.
infer measures VLAModel.forward, NOT autoregressive generation or robot latency.
data replays the stored conversation (including its reference assistant tokens).
synthetic measures a reproducible input shape, not prediction quality.
ACCs are left unavailable: this dense ANN has no separately defined spike-AC
metric; MAC-equivalents use the convention 1 MAC = 2 FLOPs. Quantized operations
are reported as dense-equivalent arithmetic, not hardware instruction counts.
Training history/energy cannot be reconstructed from an existing checkpoint.

Counting references:
https://docs.pytorch.org/docs/main/generated/torch.utils.flop_counter.FlopCounterMode.html
https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import csv


def write_report(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "report.json")
    rows = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{prefix}.{key}" if prefix else key)
        else:
            rows.append((prefix, json.dumps(value, ensure_ascii=False)
                         if isinstance(value, list) else value))

    flatten(report)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def environment():
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "psutil", "nvidia-ml-py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": sys.version,
            "packages": packages, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "command": sys.argv, "cwd": str(Path.cwd())}


class ResourceMonitor:
    """Sample process-tree RSS and selected physical GPU boards without CUDA init."""

    def __init__(self, pid, gpu_ids, interval=0.2, csv_path=None):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Install monitoring dependency: pip install psutil") from exc
        self.psutil = psutil
        self.pid, self.interval = pid, interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.errors = set()
        self.gpus = []
        self.nvml = None
        self.peak_rss = 0
        self.samples = 0
        self.csv_path = csv_path
        self.csv_file = None
        self.writer = None
        self.started = None
        self.requested_gpus = list(gpu_ids)
        if gpu_ids:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                for selector in gpu_ids:
                    try:
                        handle = (pynvml.nvmlDeviceGetHandleByIndex(int(selector))
                                  if str(selector).isdigit() else
                                  pynvml.nvmlDeviceGetHandleByUUID(str(selector)))
                        uuid = self._text(pynvml.nvmlDeviceGetUUID(handle))
                        if any(g["uuid"] == uuid for g in self.gpus):
                            continue
                        self.gpus.append({"handle": handle, "uuid": uuid,
                                          "name": self._text(pynvml.nvmlDeviceGetName(handle)),
                                          "energy_start_mj": None, "energy_end_mj": None,
                                          "power_integral_j": 0., "power_coverage_s": 0.,
                                          "last_power": None, "power_samples": 0,
                                          "peak_board_memory_bytes": 0,
                                          "peak_process_tree_memory_bytes": None,
                                          "process_observed": False})
                    except Exception as exc:
                        self.errors.add(f"GPU {selector}: {exc}")
            except Exception as exc:
                self.errors.add(f"NVML unavailable: {exc}; install nvidia-ml-py and check driver")

    @staticmethod
    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def _processes(self):
        try:
            root = self.psutil.Process(self.pid)
            processes = [root] + root.children(recursive=True)
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            return set(), 0
        rss, pids = 0, set()
        for process in processes:
            try:
                rss += process.memory_info().rss
                pids.add(process.pid)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        return pids, rss

    def _energy(self, gpu):
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(gpu["handle"])
        except Exception:
            return None

    def sample(self):
        with self.lock:
            elapsed = time.perf_counter() - self.started
            pids, rss = self._processes()
            self.peak_rss = max(self.peak_rss, rss)
            self.samples += 1
            for gpu in self.gpus:
                power, board_memory, process_memory = None, None, None
                try:
                    power = self.nvml.nvmlDeviceGetPowerUsage(gpu["handle"]) / 1000.
                    previous = gpu["last_power"]
                    if previous is not None:
                        dt = elapsed - previous[0]
                        gpu["power_integral_j"] += (previous[1] + power) * 0.5 * dt
                        gpu["power_coverage_s"] += dt
                    gpu["last_power"] = (elapsed, power)
                    gpu["power_samples"] += 1
                except Exception as exc:
                    gpu["last_power"] = None
                    self.errors.add(f"Power {gpu['uuid']}: {exc}")
                try:
                    board_memory = self.nvml.nvmlDeviceGetMemoryInfo(gpu["handle"]).used
                    gpu["peak_board_memory_bytes"] = max(gpu["peak_board_memory_bytes"], board_memory)
                except Exception as exc:
                    self.errors.add(f"VRAM {gpu['uuid']}: {exc}")
                try:
                    processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(gpu["handle"])
                    selected = [p for p in processes if p.pid in pids]
                    gpu["process_observed"] |= bool(selected)
                    values = [p.usedGpuMemory for p in selected]
                    if all(isinstance(v, int) and 0 <= v < 2**63 for v in values):
                        process_memory = sum(values)
                        gpu["peak_process_tree_memory_bytes"] = max(
                            gpu["peak_process_tree_memory_bytes"] or 0, process_memory)
                except Exception as exc:
                    self.errors.add(f"Process VRAM {gpu['uuid']}: {exc}")
                if self.writer:
                    self.writer.writerow([elapsed, rss, gpu["uuid"], power, board_memory, process_memory])
            if self.writer and not self.gpus:
                self.writer.writerow([elapsed, rss, "", "", "", ""])
            if self.csv_file:
                self.csv_file.flush()

    def start(self):
        if self.csv_path:
            self.csv_file = Path(self.csv_path).open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["elapsed_s", "tree_rss_bytes", "gpu_uuid", "board_power_w",
                                  "board_memory_bytes", "tree_gpu_memory_bytes"])
        self.started = time.perf_counter()
        for gpu in self.gpus:
            gpu["energy_start_mj"] = self._energy(gpu)
        self.sample()

        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        elapsed = time.perf_counter() - self.started
        gpu_results = []
        for gpu in self.gpus:
            start, end = gpu["energy_start_mj"], self._energy(gpu)
            energy, method = None, "unavailable"
            coverage = min(1., gpu["power_coverage_s"] / max(elapsed, 1e-9))
            complete = False
            if start is not None and end is not None and end >= start:
                energy, method = (end - start) / 1000., "NVML cumulative energy difference"
                coverage, complete = 1., True
            elif gpu["power_samples"] >= 2:
                energy, method = gpu["power_integral_j"], "sampled power trapezoidal estimate"
                complete = coverage >= 0.95
            gpu_results.append({k: v for k, v in gpu.items()
                                if k in ("uuid", "name", "power_samples", "power_coverage_s",
                                         "peak_board_memory_bytes", "peak_process_tree_memory_bytes",
                                         "process_observed")})
            gpu_results[-1].update(energy_j=energy, energy_method=method,
                                   energy_coverage_fraction=coverage,
                                   energy_window_complete=complete,
                                   energy_kwh=None if energy is None else energy / 3.6e6)
        energies = [g["energy_j"] for g in gpu_results]
        all_known = bool(energies) and all(e is not None for e in energies)
        all_known &= all(g["energy_window_complete"] for g in gpu_results)
        # A missing selected GPU must not silently produce a partial total.
        all_known &= len(gpu_results) == len(set(self.requested_gpus))
        total = sum(energies) if all_known else None
        if self.csv_file:
            self.csv_file.close()
        if self.nvml:
            self.nvml.nvmlShutdown()
        return {"elapsed_s": elapsed, "sample_interval_s": self.interval,
                "samples": self.samples, "process_tree_peak_rss_bytes": self.peak_rss,
                "requested_gpu_ids": self.requested_gpus, "gpus": gpu_results,
                "observed_process_gpu_count": sum(g["process_observed"] for g in gpu_results),
                "total_selected_gpu_energy_j": total,
                "total_selected_gpu_energy_kwh": None if total is None else total / 3.6e6,
                "energy_scope": "Selected GPU boards, including idle power and other processes; excludes CPU/RAM/PSU. No idle subtraction.",
                "ram_scope": "Sampled sum of process-tree RSS; shared pages may be counted more than once.",
                "errors": sorted(self.errors)}


def model_size(model):
    import torch
    logical, packed, trainable = 0, 0, 0
    by_component, seen, storages = {}, set(), {}
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count = parameter.numel()
        quant_state = getattr(parameter, "quant_state", None)
        if quant_state is not None and getattr(quant_state, "shape", None) is not None:
            count = math.prod(quant_state.shape)
        elif type(parameter).__name__ in ("Params4bit", "Int8Params"):
            owner = model.get_submodule(name.rsplit(".", 1)[0])
            if hasattr(owner, "in_features") and hasattr(owner, "out_features"):
                count = owner.in_features * owner.out_features
            else:
                raise RuntimeError(f"Cannot determine uncompressed parameter count: {name}")
        logical += count
        packed += parameter.numel()
        trainable += count if parameter.requires_grad else 0
        component = name.split(".")[0]
        by_component[component] = by_component.get(component, 0) + count
    state = model.state_dict()
    payload_bytes = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            payload_bytes += value.numel() * value.element_size()
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return {"logical_parameters": logical, "logical_parameters_million": logical / 1e6,
            "packed_parameter_elements": packed, "trainable_parameters_current_stage": trainable,
            "logical_parameters_by_component": by_component,
            "state_dict_tensor_payload_bytes": payload_bytes,
            "state_dict_unique_storage_bytes": sum(storages.values()),
            "state_dict_unique_storage_mib": sum(storages.values()) / 2**20,
            "note": "Logical counts expand quantized weights; storage includes state_dict quantization metadata. Payload can duplicate tied weights. Neither includes activations/optimizer/allocator."}


def tensor_info(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "elements": value.numel(), "bytes": value.numel() * value.element_size(),
                "bytes_per_sample": (value.numel() * value.element_size() / value.shape[0]
                                     if value.ndim and value.shape[0] else None)}
    if isinstance(value, (list, tuple)):
        return [tensor_info(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_info(v) for k, v in value.items()}
    return value


def feature_sizes(model, forward):
    captured, handles = {}, []
    for name in ("adapter", "visual_prefix", "shared_fpn", "mask_decoder", "pose_decoder"):
        if hasattr(model, name):
            def hook(module, inputs, output, key=name):
                captured.setdefault(key, []).append(tensor_info(output))
            handles.append(getattr(model, name).register_forward_hook(hook))
    try:
        output = forward()
        result = {"intermediate_features": captured, "outputs": tensor_info(output),
                  "definition": "No single retrieval descriptor is defined by VLAModel. Report actual adapter/prefix/spatial feature tensors per invocation; these are not all retained simultaneously."}
        del output
        return result
    finally:
        for handle in handles:
            handle.remove()


def count_operations(model, forward, batch_size):
    """Count registered PyTorch ops; supplement opaque BNB/MHA, never double count."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    class AuditedCounter(FlopCounterMode):
        def __init__(self):
            super().__init__(display=False)
            self.uncounted = Counter()

        def _count_flops(self, func_packet, out, args, kwargs):
            if func_packet not in self.flop_registry:
                self.uncounted[str(func_packet)] += 1
            return super()._count_flops(func_packet, out, args, kwargs)

    counter = AuditedCounter()
    handles, corrections, stack = [], [], {}

    def before(module, args, kwargs):
        stack.setdefault(id(module), []).append(counter.get_total_flops())

    def after(module, args, kwargs, output, name):
        observed = counter.get_total_flops() - stack[id(module)].pop()
        if isinstance(module, torch.nn.MultiheadAttention):
            query = args[0] if args else kwargs["query"]
            key = args[1] if len(args) > 1 else kwargs["key"]
            value = args[2] if len(args) > 2 else kwargs["value"]
            b = query.shape[0 if module.batch_first else 1] if query.ndim == 3 else 1
            q = query.shape[1 if module.batch_first else 0] if query.ndim == 3 else query.shape[0]
            k = key.shape[1 if module.batch_first else 0] if key.ndim == 3 else key.shape[0]
            d = module.embed_dim
            # Q/K/V projections, QK^T and AV, output projection; excludes softmax/bias.
            expected = 2 * b * (q * query.shape[-1] * d + k * key.shape[-1] * d +
                                k * value.shape[-1] * d + 2 * q * k * d + q * d * d)
        else:
            x = args[0] if args else kwargs.get("x", kwargs.get("input"))
            expected = 2 * x.numel() * module.out_features
        missing = max(0, expected - observed)
        corrections.append({"module": name, "type": type(module).__name__,
                            "dense_formula_flops": expected,
                            "already_counted_flops": observed, "added_flops": missing})

    for name, module in model.named_modules():
        is_bnb = (type(module).__module__.startswith("bitsandbytes") and
                  hasattr(module, "in_features") and hasattr(module, "out_features"))
        if is_bnb or isinstance(module, torch.nn.MultiheadAttention):
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            def hook(mod, args, kwargs, out, key=name):
                after(mod, args, kwargs, out, key)
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
    # Some PyTorch ModuleTracker versions attach backward hooks even in
    # inference_mode when an expanded Parameter view retains requires_grad.
    # Only forward is counted; restore every original flag after profiling.
    grad_flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for parameter, _ in grad_flags:
            parameter.requires_grad_(False)
        with counter:
            output = forward()
            del output
        supported = counter.get_total_flops()
        added = sum(c["added_flops"] for c in corrections)
        total = supported + added
        return {"status": "partial_operator_coverage", "counted_flops_per_batch": supported,
                "dense_equivalent_correction_flops": added,
                "dense_equivalent_flops_per_batch": total,
                "dense_equivalent_gflops_per_sample": total / batch_size / 1e9,
                "mac_equivalents_per_sample": total / batch_size / 2,
                "gmac_equivalents_per_sample": total / batch_size / 2e9,
                "accs": None,
                "accs_reason": "No separate spike-accumulation metric or hardware instruction counter; ACCs are not inferred from MACs.",
                "scope": "Registered matmul/conv/attention arithmetic plus opaque BNB/MHA dense formulas. 1 MAC=2 FLOPs. Not exhaustive scalar operations or physical 4-bit instructions. Inspect uncounted_ops before claiming total FLOPs.",
                "registered_ops": {str(k): v for k, v in counter.flop_counts.get("Global", {}).items()},
                "uncounted_ops": dict(counter.uncounted), "module_corrections": corrections}
    finally:
        for handle in handles:
            handle.remove()
        for parameter, requires_grad in grad_flags:
            parameter.requires_grad_(requires_grad)


def percentile(values, percent):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100.
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def benchmark_forward(forward, synchronize, iterations, warmup, batch_size):
    for _ in range(warmup):
        output = forward()
        del output
    synchronize()
    times = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        output = forward()
        synchronize()
        times.append(time.perf_counter() - start)
        del output
    # A separate window, no per-request synchronization, for steady throughput.
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = forward()
        del output
    synchronize()
    elapsed = time.perf_counter() - start
    mean = statistics.mean(times)
    return {"iterations_per_window": iterations, "warmup_iterations": warmup,
            "batch_size": batch_size, "batch_latency_mean_ms": mean * 1000,
            "batch_latency_median_ms": statistics.median(times) * 1000,
            "batch_latency_p90_ms": percentile(times, 90) * 1000,
            "batch_latency_p95_ms": percentile(times, 95) * 1000,
            "batch_latency_std_ms": statistics.pstdev(times) * 1000,
            "amortized_latency_ms_per_sample": mean * 1000 / batch_size,
            "synchronized_samples_per_second": batch_size / mean,
            "steady_throughput_samples_per_second": iterations * batch_size / elapsed,
            "fps": iterations * batch_size / elapsed,
            "throughput_window_s": elapsed,
            "latency_samples_ms": [t * 1000 for t in times],
            "scope": "Repeated preloaded batch, model.forward only, GPU synchronization at timing boundaries. Includes host dispatch; excludes preprocessing/H2D/D2H/postprocessing/generation. FPS counts images, not tokens. Amortized latency is not single-request latency when B>1."}


def input_batch(args, cfg, model):
    import torch
    keys = ("pixel_values_dino", "pixel_values_siglip", "input_ids", "attention_mask")
    if args.input == "data":
        from dataset import VLADataset
        dataset = VLADataset(
            jsonl_paths=[args.jsonl or cfg.test_jsonl], data_root=cfg.data_root,
            tokenizer=model.tokenizer, max_length=args.seq_len,
            dino_size=cfg.dino_size, siglip_size=cfg.siglip_size, mask_size=cfg.img_size,
            augment=False, exclude_classes=cfg.exclude_classes, strict_data=True)
        end = args.sample_index + args.batch_size
        if end > len(dataset):
            raise ValueError("Selected real batch extends beyond dataset")
        samples = [dataset[i] for i in range(args.sample_index, end)]
        batch = {key: torch.stack([s[key] for s in samples]) for key in keys}
        source = {"type": "reference_conversation_replay", "jsonl": args.jsonl or cfg.test_jsonl,
                  "sample_index": args.sample_index, "task_argument_applies": False}
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        candidates = model.tokenizer.encode("object", add_special_tokens=False)
        filler = next((i for i in candidates if i not in (model.mask_token_id, model.pose_token_id)), None)
        if filler is None:
            raise ValueError("Tokenizer produced no ordinary filler token")
        ids = torch.full((args.batch_size, args.seq_len), filler, dtype=torch.long)
        if args.task in ("mask", "both"):
            ids[:, -2] = model.mask_token_id
        if args.task in ("pose", "both"):
            ids[:, -1] = model.pose_token_id
        batch = {"pixel_values_dino": torch.randn(args.batch_size, 3, cfg.dino_size, cfg.dino_size, generator=g),
                 "pixel_values_siglip": torch.randn(args.batch_size, 3, cfg.siglip_size, cfg.siglip_size, generator=g),
                 "input_ids": ids, "attention_mask": torch.ones_like(ids)}
        source = {"type": "synthetic_fixed_shape", "task": args.task}
    source.update(tensors=tensor_info(batch),
                  active_text_tokens_per_sample=batch["attention_mask"].sum(-1).tolist(),
                  mask_token_present_per_sample=(batch["input_ids"] == model.mask_token_id).any(-1).tolist(),
                  pose_token_present_per_sample=(batch["input_ids"] == model.pose_token_id).any(-1).tolist(),
                  labels_or_ground_truth_passed_to_forward=False)
    return batch, source


def infer(args, report):
    import torch
    import yaml
    from checkpoint_compat import load_evaluation_checkpoint
    from train import VLAConfig, configure_stage, set_global_seed
    from model import build_vla_model

    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run inference benchmark as a single process, not with torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model benchmark requires a CUDA server with the training dependencies")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use --device cuda:N for the full model")
    torch.cuda.set_device(device)
    cfg = VLAConfig()
    config_values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    unknown = set(config_values) - set(VLAConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for key, value in config_values.items():
        setattr(cfg, key, value)
    set_global_seed(cfg.seed)
    properties = torch.cuda.get_device_properties(device)
    # CUDA UUID resolves physical NVML board even with CUDA_VISIBLE_DEVICES remapping.
    uuid = getattr(properties, "uuid", None)
    gpu_ids = [str(uuid)] if uuid is not None else []
    report.update(config=asdict(cfg), device={"cuda_index": torch.cuda.current_device(),
                  "name": properties.name, "total_memory_bytes": properties.total_memory,
                  "uuid": str(uuid) if uuid is not None else None,
                  "cuda_runtime": torch.version.cuda},
                  precision={"autocast": "bfloat16", "llm_4bit": cfg.use_4bit_quant},
                  training_cost={"status": "not_measured", "reason": "Run monitor around the actual training command; checkpoint alone has no reliable total time/energy."})
    monitor = ResourceMonitor(os.getpid(), gpu_ids, args.sample_interval,
                              Path(args.output) / "load_resources.csv").start()
    try:
        model = build_vla_model(cfg, device=device)
        # Match train's placement: quantized LLM is already device-mapped;
        # place the CPU-created vision/fusion/decoder parameters without casting.
        for parameter in model.parameters():
            if parameter.device.type == "cpu":
                parameter.data = parameter.data.to(device)
        for buffer in model.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.to(device)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint, incompatible, restoration = load_evaluation_checkpoint(
            checkpoint_path, model, strict=not args.allow_nonstrict,
        )
        report["checkpoint"] = {"path": str(checkpoint_path.resolve()),
                                "file_bytes": checkpoint_path.stat().st_size,
                                "strict": not args.allow_nonstrict,
                                "missing_keys": list(incompatible.missing_keys),
                                "unexpected_keys": list(incompatible.unexpected_keys),
                                "quantization_restoration": restoration,
                                "file_size_note": "Training checkpoint may include optimizer/scheduler; not model-only size."}
        del checkpoint
        gc.collect()
        model.eval()
        stage_counts = {}
        for stage in (1, 2):
            configure_stage(model, stage, rank=1)
            stage_counts[str(stage)] = model_size(model)["trainable_parameters_current_stage"]
        report["model_size"] = model_size(model)
        report["model_size"]["trainable_parameters_by_stage"] = stage_counts
        # Explicitly disable checkpointing for inference only; no training changes.
        if hasattr(model.llm, "gradient_checkpointing_disable"):
            model.llm.gradient_checkpointing_disable()
        batch, report["input"] = input_batch(args, cfg, model)
        batch = {k: v.to(device) for k, v in batch.items()}
        torch.cuda.synchronize(device)
    finally:
        report["load_resources"] = monitor.stop()
        write_report(args.output, report)

    def forward():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(**batch, stage=2)

    synchronize = lambda: torch.cuda.synchronize(device)
    with torch.inference_mode():
        # Untimed feature inspection also initializes lazy model caches.
        report["feature_and_output_sizes"] = feature_sizes(model, forward)
        for _ in range(args.warmup):
            output = forward()
            del output
        synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        monitor = ResourceMonitor(os.getpid(), gpu_ids, args.sample_interval,
                                  Path(args.output) / "inference_resources.csv").start()
        try:
            report["speed"] = benchmark_forward(forward, synchronize, args.iterations, 0, args.batch_size)
            report["speed"]["warmup_iterations"] = args.warmup
        finally:
            report["inference_resources"] = monitor.stop()
        report["cuda_memory"] = {
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline_allocated,
            "scope": "Warmed latency and throughput windows, excluding FLOP profiling; baseline includes model, input and retained caches. PyTorch allocation excludes driver/non-PyTorch memory."}
        write_report(args.output, report)
        if args.skip_flops:
            report["compute"] = {"status": "skipped"}
        else:
            try:
                report["compute"] = count_operations(model, forward, args.batch_size)
            except Exception as exc:
                report["compute"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    write_report(args.output, report)
    print(json.dumps({"parameters_M": report["model_size"]["logical_parameters_million"],
                      "weight_storage_MiB": report["model_size"]["state_dict_unique_storage_mib"],
                      "GFLOPs_per_sample": report["compute"].get("dense_equivalent_gflops_per_sample"),
                      "latency_ms": report["speed"]["batch_latency_mean_ms"],
                      "FPS": report["speed"]["fps"],
                      "peak_GPU_MiB": report["cuda_memory"]["peak_allocated_bytes"] / 2**20}, indent=2))
    return 0


def monitor_training(args, report):
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("Provide the original training command after --")
    selectors = [] if args.gpu_ids == "none" else [s.strip() for s in args.gpu_ids.split(",") if s.strip()]
    # Monitoring the wrapper tree includes the child launcher and every DDP worker.
    monitor = ResourceMonitor(os.getpid(), selectors, args.sample_interval,
                              Path(args.output) / "training_resources.csv").start()
    child = None
    interrupted = False
    start = time.perf_counter()
    try:
        child = subprocess.Popen(command, start_new_session=(os.name != "nt"))
        code = child.wait()
    except KeyboardInterrupt:
        interrupted = True
        if child is not None:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGINT)
            else:
                child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
        code = 130
    finally:
        elapsed = time.perf_counter() - start
        report["resources"] = monitor.stop()
        report["training_cost"] = {"command": command, "wall_time_s": elapsed,
                                   "wall_time_hours": elapsed / 3600.,
                                   "selected_gpu_count": len(set(selectors)),
                                   "selected_gpu_hours": elapsed * len(set(selectors)) / 3600.,
                                   "scope": "Complete launched command including initialization/data/validation/checkpoint I/O; selected GPU-hours are reservation cost, not measured utilization.",
                                   "historical_checkpoint_cost": "not reconstructible"}
        write_report(args.output, report)
    report["training_cost"].update(exit_code=code, interrupted=interrupted,
                                   completed_successfully=(code == 0))
    write_report(args.output, report)
    print(json.dumps(report["training_cost"], indent=2, ensure_ascii=False))
    return code if code >= 0 else 128 - code


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True, help="Report directory (report.json, summary.csv, resource CSVs)")
    common.add_argument("--sample-interval", type=float, default=0.2)
    p = sub.add_parser("infer", parents=[common], help="Full checkpoint model-forward benchmark")
    p.add_argument("--config", default="config_full.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--input", choices=["synthetic", "data"], default="synthetic")
    p.add_argument("--task", choices=["both", "mask", "pose", "vqa"], default="both",
                   help="Synthetic input only; data uses the record's original task tokens")
    p.add_argument("--jsonl", help="Real-data input, default cfg.test_jsonl")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--seq-len", type=positive_int, default=128,
                   help="Text length; visual tokens are prepended by VLAModel")
    p.add_argument("--batch-size", type=positive_int, default=1)
    p.add_argument("--iterations", type=positive_int, default=100)
    p.add_argument("--warmup", type=positive_int, default=10)
    p.add_argument("--allow-nonstrict", action="store_true", help="Explicitly allow checkpoint key mismatches; report them")
    p.add_argument("--skip-flops", action="store_true")
    p = sub.add_parser("monitor", parents=[common], help="Wrap the unchanged real training command")
    p.add_argument("--gpu-ids", required=True, help="Physical NVML indices/UUIDs, e.g. 0,1; none for CPU-only")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_interval) or args.sample_interval < 0.05:
        parser.error("--sample-interval must be finite and >= 0.05 seconds")
    if args.mode == "infer" and (args.seq_len < 2 or args.sample_index < 0):
        parser.error("--seq-len must be >= 2 and --sample-index must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "mode": args.mode, "environment": environment(),
              "status": "running", "units": "bytes, seconds, joules; MiB=2^20 bytes; 1 kWh=3.6e6 J"}
    write_report(args.output, report)
    try:
        code = infer(args, report) if args.mode == "infer" else monitor_training(args, report)
        report["status"] = "completed" if code == 0 else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    print(f"Reports: {Path(args.output).resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
