#!/usr/bin/env python3
"""Paper-aligned two-stage training for TRP-Grasp.

Stage 1 freezes DINOv2, SigLIP and Qwen2 and trains the newly introduced
multimodal/spatial modules. Stage 2 keeps both visual backbones frozen and
opens only Qwen2 LoRA parameters together with the Stage-1 modules.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent))
from model import VLAModel, build_vla_model
from dataset import build_dataloaders, ensure_special_tokens
from utils import (
    build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint,
    compute_miou, compute_iou_components, compute_pose_error,
    compute_mean_iou,
    mask_loss_dynamic, uncertainty_weighted_loss,
    prefix_contrastive_loss,
    paper_pose_loss,
    compute_beta_target,
    MetricTracker,
    #  新增
    gaussian_heatmap_loss,
    apply_pcgrad_post_hook, collect_task_gradients_hook,  # PCGrad
    viewpoint_weak_label_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VLAConfig:
    # Paths
    data_root: str = "./data/LabGrasp-Deploy-Real"
    train_jsonl: str = "./data/LabGrasp-Deploy-Real/annotations/train.jsonl"
    val_jsonl: str = "./data/LabGrasp-Deploy-Real/annotations/val.jsonl"
    test_jsonl: str = "./data/LabGrasp-Deploy-Real/annotations/test.jsonl"
    output_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs/logs"
    resume_from: str = ""

    # Model
    llm_name_or_path: str = "./pretrained/qwen2-7b"
    dinov2_name: str = "./pretrained/dinov2-large"
    siglip_name: str = "./pretrained/siglip"
    llm_dim: int = 3584
    dinov2_dim: int = 1024
    siglip_dim: int = 1152
    vit_fused_dim: int = 1024
    img_size: int = 336
    dino_size: int = 336
    siglip_size: int = 384
    use_4bit_quant: bool = True

    # Architecture
    num_queries: int = 64
    fpn_channels: int = 256
    spatial_beta: float = 2.0
    num_visual_prefix: int = 4

    # [ABLATION] module switches
    use_capd: bool = True
    use_vaf: bool = True
    use_shared_decoder: bool = True
    vaf_fixed_gate: Optional[float] = None
    capd_use_confidence_weighting: bool = True
    capd_use_progressive_sampling: bool = True
    capd_fixed_alpha: float = 0.9
    capd_window_size: int = 5

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_lr: float = 3e-5

    # Data
    use_combined_jsonl: bool = True
    exclude_classes: tuple = ("pipette",)
    strict_data: bool = True
    max_seq_len: int = 512
    num_workers: int = 8

    # Training — [MEM] batch_size=1 for 32GB VRAM safety
    stage1_epochs: int = 3
    stage2_epochs: int = 6
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16

    # Optimizer
    learning_rate: float = 3e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03

    # Loss
    lambda_text: float = 1.0
    lambda_mask: float = 1.0
    lambda_pose: float = 0.5
    smooth_l1_beta: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_ramp_steps: int = 4000
    area_weight_min: float = 0.5
    area_weight_max: float = 5.0
    area_reference_ratio: float = 0.05
    prefix_contrastive_weight: float = 0.0
    lambda_heatmap: float = 1.0
    heatmap_fg_weight: float = 5.0
    lambda_consist: float = 1.0
    lambda_vp: float = 0.1
    capd_anneal_warmup: float = 0.1
    capd_anneal_end: float = 0.7
    capd_alpha_end: float = 0.9

    # [OPT-15] PCGrad
    use_pcgrad: bool = False
    pcgrad_interval: int = 4  # Apply PCGrad every N accum steps (saves memory)

    # Logging
    log_every_n_steps: int = 20
    eval_every_n_steps: int = 500
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    if "RANK" not in os.environ:
        rank = world_size = local_rank = 0
        torch.cuda.set_device(0)
        return rank, world_size, local_rank
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def set_global_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_training_paths(cfg):
    required_files = {
        "train_jsonl": cfg.train_jsonl,
        "val_jsonl": cfg.val_jsonl,
        "test_jsonl": cfg.test_jsonl,
    }
    missing = [f"{name}={path}" for name, path in required_files.items()
               if not path or not os.path.isfile(path)]
    if not os.path.isdir(cfg.data_root):
        missing.append(f"data_root={cfg.data_root}")
    for name in ("llm_name_or_path", "dinov2_name", "siglip_name"):
        path = getattr(cfg, name)
        if not os.path.exists(path):
            missing.append(f"{name}={path}")
    if missing:
        raise FileNotFoundError("Missing required training resource(s):\n  - " +
                                "\n  - ".join(missing))
    resolved = {name: os.path.realpath(path) for name, path in required_files.items()}
    if len(set(resolved.values())) != 3:
        raise ValueError(f"train/val/test JSONL paths must be distinct: {resolved}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage freeze/unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def configure_stage(model: VLAModel, stage: int, rank: int):
    decoders = ([model.shared_fpn] if model.use_shared_decoder else
                [model.mask_decoder, model.pose_decoder])
    for p in model.vision_encoder.dinov2.parameters():
        p.requires_grad_(False)
    for p in model.vision_encoder.siglip.parameters():
        p.requires_grad_(False)

    if stage == 1:
        for p in model.llm.parameters():
            p.requires_grad_(False)
        for mod in [model.adapter, *decoders, model.mask_head,
                    model.pose_head, model.vision_encoder.va_fusion,
                    model.visual_prefix]:
            for p in mod.parameters():
                p.requires_grad_(True)
        for p in model.vision_encoder.level_fusion.parameters():
            p.requires_grad_(True)

    elif stage == 2:
        for name, p in model.llm.named_parameters():
            # Paper Sec. 3.5.2: Qwen2 is adapted through LoRA; the pretrained
            # embedding table and full LM head remain frozen.
            p.requires_grad_("lora_" in name)
        for mod in [model.adapter, *decoders, model.mask_head,
                    model.pose_head, model.vision_encoder.va_fusion,
                    model.visual_prefix]:
            for p in mod.parameters():
                p.requires_grad_(True)
        for p in model.vision_encoder.level_fusion.parameters():
            p.requires_grad_(True)
        # Paper Sec. 3.5.2: both visual backbones remain frozen in Stage 2.
        for p in model.vision_encoder.dinov2.parameters():
            p.requires_grad_(False)
        for p in model.vision_encoder.siglip.parameters():
            p.requires_grad_(False)

    if is_main(rank):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(
            f"[Stage {stage}] Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
            f"({100 * trainable / total:.2f}%)"
        )


def get_shared_params_dict(raw_model: nn.Module) -> dict:
    """Legacy optional PCGrad scope; disabled in paper-aligned configs."""
    shared = {}
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if "shared_fpn" in name or "adapter" in name:
            shared[name] = p
    return shared


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_step(model, batch, device, stage, global_step, cfg, log_vars,
                   shared_params_dict=None, total_steps=1,
                   grad_accum_divisor=None):
    """Compute one heterogeneous batch of the paper objective (Eq. 31-37)."""
    pv_dino   = batch["pixel_values_dino"].to(device, non_blocking=True)
    pv_siglip = batch["pixel_values_siglip"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels    = batch["labels"].to(device, non_blocking=True)
    gt_mask   = batch["gt_mask"].to(device, non_blocking=True)
    gt_pose   = batch["gt_pose"].to(device, non_blocking=True)
    has_mask  = batch["has_mask"].to(device, non_blocking=True)
    has_pose  = batch["has_pose"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            pixel_values_dino=pv_dino, pixel_values_siglip=pv_siglip,
            input_ids=input_ids, attention_mask=attn_mask,
            labels=labels, gt_mask=gt_mask, gt_pose=gt_pose, stage=stage,
            global_step=global_step, total_steps=total_steps,   # [CAPD]
        )

        loss_ce = outputs.get("loss_ce")
        loss_mask = loss_pose = loss_heatmap = None

        # ── Mask loss ───────────────────────────────────────
        if "heatmap_logits" in outputs and has_mask.any():
            hmap = outputs["heatmap_logits"]
            mdev = hmap.device
            vidx = has_mask.to(mdev).nonzero(as_tuple=True)[0]
            if len(vidx) > 0:
                gt_m = F.interpolate(
                    gt_mask.float().to(mdev)[vidx],
                    size=hmap.shape[-2:], mode='nearest',
                )
                loss_mask = mask_loss_dynamic(
                    hmap[vidx], gt_m, global_step, cfg.dice_ramp_steps,
                    cfg.focal_alpha, cfg.focal_gamma,
                    cfg.area_weight_min, cfg.area_weight_max,
                    cfg.area_reference_ratio,
                )

        # ── [Heatmap] Gaussian heatmap loss for grasp center supervision ──
        if "grasp_heatmap" in outputs and has_pose.any():
            grasp_logits = outputs["grasp_heatmap"]
            gdev = grasp_logits.device
            vidx_p = has_pose.to(gdev).nonzero(as_tuple=True)[0]
            if len(vidx_p) > 0:
                # gt_pose[:, :2] = (x_norm, y_norm) 即 GT grasp 中心
                gt_centers = gt_pose.to(gdev)[vidx_p][:, :2]  # [Bv, 2]
                # σ 与特征图尺度匹配: 336 → 6, 84 → 1.5
                H_grasp = grasp_logits.shape[-1]
                sigma_px = max(2.0, H_grasp * 0.018)
                loss_heatmap = gaussian_heatmap_loss(
                    grasp_logits[vidx_p], gt_centers,
                    sigma_px=sigma_px,
                    fg_weight=getattr(cfg, "heatmap_fg_weight", 5.0),
                )

        # ── Paper Eq. (28)-(30): xyw Smooth-L1 + double-angle unit-vector loss ──
        if "pred_pose" in outputs and has_pose.any():
            pp = outputs["pred_pose"]
            pdev = pp.device
            vidx = has_pose.to(pdev).nonzero(as_tuple=True)[0]
            if len(vidx) > 0:
                loss_pose = paper_pose_loss(
                    pp[vidx],
                    gt_pose.to(pdev)[vidx],
                    smooth_l1_beta=cfg.smooth_l1_beta,
                )

        # [OPT-10] Prefix contrastive loss (强化版本, margin=1.0 可在配置)
        loss_prefix = None
        if (getattr(cfg, "prefix_contrastive_weight", 0.0) > 0
                and "prefix_features" in outputs):
            loss_prefix = prefix_contrastive_loss(
                outputs["prefix_features"].to(device)
            ) * cfg.prefix_contrastive_weight

        # Apply lambda weights
        if loss_ce is not None:
            loss_ce = loss_ce * cfg.lambda_text
        if loss_mask is not None:
            loss_mask = loss_mask * cfg.lambda_mask
        if loss_pose is not None:
            loss_pose = loss_pose * cfg.lambda_pose
        if loss_heatmap is not None:
            loss_heatmap = loss_heatmap * getattr(cfg, "lambda_heatmap", 1.0)

        # [Heatmap] heatmap loss 走独立通道, 不并入 mask uncertainty
        # ─────────────────────────────────────────────────────
        # 原 attempt1 bug: loss_mask = loss_mask + loss_heatmap 让 σ_m 同时拟合两个损失
        #   heatmap 在 fg_weight 加权下数值小且容易优化 → 持续拉低 σ_m
        # → mask 通道权重 1/(2σ²) 误判性上升 → mask 任务过度自信 → train/test gap 拉大
        # 修复: heatmap 用固定权重直接加到 total_loss, 与 mask uncertainty 完全解耦
        # 显存: 此改动有 ~1.5GB 增量, 已通过 [Memory] 冻结 ViT 省出 3-5GB 覆盖

        total_loss, loss_detail = uncertainty_weighted_loss(
            loss_ce, loss_mask, loss_pose,
            log_vars["text"], log_vars["mask"], log_vars["pose"],
        )

        # [Heatmap] heatmap 固定权重叠加, 不走 uncertainty
        if loss_heatmap is not None:
            total_loss = total_loss + loss_heatmap

        if loss_prefix is not None:
            total_loss = total_loss + loss_prefix
            loss_detail["loss_prefix"] = loss_prefix.item()

        if loss_heatmap is not None:
            loss_detail["loss_heatmap"] = loss_heatmap.item() if isinstance(loss_heatmap, torch.Tensor) else loss_heatmap

        # ── [CAPD] Centroid Consistency Loss ──────────────
        # 退火越深 (α 越大、越依赖预测质心), 一致性约束越重要
        if "pred_center" in outputs and has_pose.any():
            pdev = outputs["pred_center"].device
            vidx_c = has_pose.to(pdev).nonzero(as_tuple=True)[0]
            if len(vidx_c) > 0:
                gt_centers_c = gt_pose.to(pdev)[vidx_c][:, :2]   # GT 中心
                pred_centers_c = outputs["pred_center"][vidx_c]
                # 退火系数 α(t)
                aw = getattr(cfg, "capd_anneal_warmup", 0.1)
                ae = getattr(cfg, "capd_anneal_end", 0.7)
                a_end = getattr(cfg, "capd_alpha_end", 0.9)
                t_warm = aw * total_steps
                t_end = ae * total_steps
                if global_step <= t_warm:
                    alpha_t = 0.0
                elif global_step >= t_end:
                    alpha_t = a_end
                else:
                    alpha_t = (global_step - t_warm) / max(t_end - t_warm, 1.0) * a_end
                lam_consist = getattr(cfg, "lambda_consist", 1.0) * alpha_t
                if lam_consist > 0:
                    mse_c = ((pred_centers_c - gt_centers_c) ** 2).sum(dim=-1).mean()
                    loss_consist = lam_consist * mse_c
                    total_loss = total_loss + loss_consist
                    loss_detail["loss_consist"] = loss_consist.item()
                    loss_detail["capd_alpha"] = alpha_t

        # ── [VA-Fusion] 类别弱监督视角损失 ─────────────────
        # 长宽比自监督已被 verify_aspect_ratio.py 实测证伪 (准确率 60%,
        # 因为 GRASP_MASK 编码抓取策略而非视角), 改用类别派生弱标签:
        # 训练时 object_class→视角 (100% 准确), 推理时门控纯靠图像预测.
        if (getattr(cfg, "use_vaf", True)
                and "vp_logit" in outputs and "view_label" in batch):
            vdev = outputs["vp_logit"].device
            view_labels = batch["view_label"].to(vdev)
            loss_vp = viewpoint_weak_label_loss(outputs["vp_logit"], view_labels)
            if loss_vp is not None:
                loss_vp = loss_vp * getattr(cfg, "lambda_vp", 0.1)
                total_loss = total_loss + loss_vp
                loss_detail["loss_vp"] = loss_vp.item()
                loss_detail["vp_gate"] = outputs["vp_gate"].mean().item()
                # 门控分类准确率 (实时监控视角推断学习进度)
                with torch.no_grad():
                    valid = view_labels >= 0
                    if valid.any():
                        pred = (outputs["vp_gate"].squeeze(-1)[valid] > 0.5).long()
                        loss_detail["vp_acc"] = (pred == view_labels[valid]).float().mean().item()

        if "beta" in outputs:
            loss_detail["beta"] = outputs["beta"].item() if isinstance(outputs["beta"], torch.Tensor) else outputs["beta"]

    # ─────────────────────────────────────────────────
    # [PCGrad] 梯度策略选择: 真 PCGrad vs 标准 backward
    # ─────────────────────────────────────────────────
    grad_divisor = grad_accum_divisor or cfg.grad_accum_steps
    use_real_pcgrad = (
        cfg.use_pcgrad
        and shared_params_dict is not None
        and len(shared_params_dict) > 0
        and global_step % cfg.pcgrad_interval == 0
        and stage == 2  # 仅 Stage 2 启用 (Stage 1 无 LLM 冲突)
    )
    
    if use_real_pcgrad:
        # [PCGrad] 真 PCGrad 与 DDP 兼容版本
        # 流程:
        #   1. autograd.grad 收集各任务对 shared params 的梯度（不污染 .grad,
        #      不污染非 shared 参数, retain_graph 让最后的 backward 还能用）
        #   2. 标准 (total_loss).backward() 让 DDP 正常 allreduce 全部参数
        #   3. apply_pcgrad_post_hook 用收集的任务梯度做投影, 覆盖 shared.grad
        #   4. 多卡时, PCGrad 后的 shared.grad 需要再做一次 all_reduce 同步
        #      (因为 DDP backward 中 allreduce 的是 *原始* shared.grad,
        #       被覆盖后各 rank 的值不一致, 必须再同步一次)
        task_grad_buffers = {}
        task_losses = {}
        if loss_mask is not None:
            task_losses["mask"] = loss_mask
        if loss_pose is not None:
            task_losses["pose"] = loss_pose
        if loss_ce is not None:
            task_losses["text"] = loss_ce
        
        if len(task_losses) >= 2:
            # 1. 收集各任务对 shared params 的梯度（不污染 .grad）
            for tname, tloss in task_losses.items():
                collect_task_gradients_hook(
                    tloss, tname, shared_params_dict, task_grad_buffers,
                    grad_accum_steps=grad_divisor,
                )
            
            # 2. 标准 backward (DDP 正常 allreduce, 最后一次会释放图)
            (total_loss / grad_divisor).backward()
            
            # 3. PCGrad 投影并覆盖 shared params 的 .grad
            if len(task_grad_buffers) >= 2:
                pcgrad_info = apply_pcgrad_post_hook(task_grad_buffers, shared_params_dict)
                loss_detail["pcgrad_conflict_rate"] = pcgrad_info.get("conflict_rate", 0.0)
                
                # 4. 多卡同步 PCGrad 后的 shared grad
                if dist.is_initialized() and dist.get_world_size() > 1:
                    for p in shared_params_dict.values():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        else:
            # 只有一个任务, 退化为标准 backward
            (total_loss / grad_divisor).backward()
    else:
        # 标准 backward
        (total_loss / grad_divisor).backward()

    return loss_detail


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — [OPT-18] Now includes mIoU and soft_iou
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, cfg, stage, global_step, log_vars,
             split_name="Test", max_batches=80):
    model.eval()

    tracker = MetricTracker([
        "loss_total", "loss_ce", "loss_mask", "loss_pose",
        "ciou", "ciou_fixed", "miou", "soft_iou",
        "translation_error_px", "angle_error_deg", "width_error_px",
    ])
    global_inter = 0.0
    global_union = 0.0

    for i, batch in enumerate(val_loader):
        # ``None`` means full evaluation. Validation is deliberately not
        # truncated because LabGrasp JSONL records are grouped by task type;
        # taking only the first N batches can omit PR or VQA completely.
        if max_batches is not None and i >= max_batches:
            break

        pv_dino   = batch["pixel_values_dino"].to(device)
        pv_siglip = batch["pixel_values_siglip"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        gt_mask   = batch["gt_mask"].to(device)
        gt_pose   = batch["gt_pose"].to(device)
        has_mask  = batch["has_mask"].to(device)
        has_pose  = batch["has_pose"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                pixel_values_dino=pv_dino, pixel_values_siglip=pv_siglip,
                input_ids=input_ids, attention_mask=attn_mask,
                labels=labels, gt_mask=gt_mask, gt_pose=gt_pose, stage=stage,
            )

            loss_ce = outputs.get("loss_ce")
            loss_mask = loss_pose = None

            if "heatmap_logits" in outputs and has_mask.any():
                hmap = outputs["heatmap_logits"]
                vidx = has_mask.to(hmap.device).nonzero(as_tuple=True)[0]
                if len(vidx) > 0:
                    gt_m = F.interpolate(
                        gt_mask.float().to(hmap.device)[vidx],
                        size=hmap.shape[-2:], mode='nearest',
                    )
                    loss_mask = mask_loss_dynamic(
                        hmap[vidx], gt_m, global_step, cfg.dice_ramp_steps,
                        cfg.focal_alpha, cfg.focal_gamma,
                        cfg.area_weight_min, cfg.area_weight_max,
                        cfg.area_reference_ratio,
                    )

            if "pred_pose" in outputs and has_pose.any():
                pp = outputs["pred_pose"]
                vidx = has_pose.to(pp.device).nonzero(as_tuple=True)[0]
                if len(vidx) > 0:
                    # Use the same paper pose objective in training and validation.
                    loss_pose = paper_pose_loss(
                        pp[vidx],
                        gt_pose.to(pp.device)[vidx],
                        smooth_l1_beta=cfg.smooth_l1_beta,
                    )

            # Apply lambda weights (consistent with training)
            if loss_ce is not None:
                loss_ce = loss_ce * cfg.lambda_text
            if loss_mask is not None:
                loss_mask = loss_mask * cfg.lambda_mask
            if loss_pose is not None:
                loss_pose = loss_pose * cfg.lambda_pose

            _, detail = uncertainty_weighted_loss(
                loss_ce, loss_mask, loss_pose,
                log_vars["text"], log_vars["mask"], log_vars["pose"],
            )

        tracker.update(detail, n=pv_dino.shape[0])

        # [OPT-18] Compute all IoU variants: cIoU(otsu), cIoU(0.5), mIoU, gIoU, soft_iou
        if "heatmap_logits" in outputs and has_mask.any():
            hmap = outputs["heatmap_logits"]
            vidx = has_mask.to(hmap.device).nonzero(as_tuple=True)[0]
            if len(vidx) > 0:
                iou_comp = compute_iou_components(
                    hmap[vidx], gt_mask.to(hmap.device)[vidx], use_otsu=True
                )
                tracker.update({"ciou": iou_comp["ciou"]}, n=len(vidx))
                tracker.update({"soft_iou": iou_comp["soft_iou"]}, n=len(vidx))
                global_inter += iou_comp["inter_sum"]
                global_union += iou_comp["union_sum"]

                # cIoU with fixed 0.5 threshold
                ciou_fixed = compute_miou(
                    hmap[vidx], gt_mask.to(hmap.device)[vidx],
                    threshold=0.5, use_otsu=False,
                )
                tracker.update({"ciou_fixed": ciou_fixed}, n=len(vidx))

                # [OPT-18] mIoU
                miou_val = compute_mean_iou(
                    hmap[vidx], gt_mask.to(hmap.device)[vidx], use_otsu=True,
                )
                tracker.update({"miou": miou_val}, n=len(vidx))

        if "pred_pose" in outputs and has_pose.any():
            pp = outputs["pred_pose"]
            vidx = has_pose.to(pp.device).nonzero(as_tuple=True)[0]
            if len(vidx) > 0:
                pose_err = compute_pose_error(pp[vidx], gt_pose.to(pp.device)[vidx], cfg.img_size)
                tracker.update(pose_err, n=len(vidx))

        # [MEM] Aggressive cleanup
        del outputs
        if i % 20 == 0:
            torch.cuda.empty_cache()

    metric_keys = [
        "loss_total", "loss_ce", "loss_mask", "loss_pose",
        "ciou", "ciou_fixed", "miou", "soft_iou",
        "translation_error_px", "angle_error_deg", "width_error_px",
    ]
    sums = torch.tensor(
        [tracker.meters[k].sum for k in metric_keys]
        + [tracker.meters[k].count for k in metric_keys]
        + [global_inter, global_union],
        device=device, dtype=torch.float64,
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    n_metrics = len(metric_keys)
    avgs = {}
    for idx, key in enumerate(metric_keys):
        count = sums[n_metrics + idx].item()
        avgs[key] = (sums[idx] / count).item() if count > 0 else float("nan")
    reduced_inter, reduced_union = sums[-2:].tolist()
    avgs["giou"] = (
        (reduced_inter + 1e-6) / (reduced_union + 1e-6)
        if reduced_union > 0 else 0.0
    )
    avgs["split"] = split_name
    model.train()
    return avgs


# ─────────────────────────────────────────────────────────────────────────────
# Stage loop
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(
    stage, model, raw_model, train_loader, val_loader,
    cfg, device, rank, writer, log_vars,
    global_step=0, best_miou=0.0, resume_ckpt=None,
):
    n_epochs = cfg.stage1_epochs if stage == 1 else cfg.stage2_epochs
    configure_stage(raw_model, stage, rank)

    optimizer = build_optimizer(raw_model, cfg, stage)
    optimizer.add_param_group({
        "params": list(log_vars.values()),
        "weight_decay": 0.0,
        "lr": cfg.learning_rate,
    })

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = n_epochs * steps_per_epoch
    schedule_total_steps = (
        (cfg.stage1_epochs + cfg.stage2_epochs)
        * steps_per_epoch
    )
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                if is_main(rank):
                    logging.info("Optimizer state restored.")
            except Exception as e:
                if is_main(rank):
                    logging.warning(f"Optimizer restore failed: {e}")
        if "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
                if is_main(rank):
                    logging.info("Scheduler state restored.")
            except Exception as e:
                if is_main(rank):
                    logging.warning(f"Scheduler restore failed: {e}")

    all_trainable = [p for p in raw_model.parameters() if p.requires_grad] + list(log_vars.values())
    # [PCGrad] 仅 Stage 2 启用真 PCGrad (Stage 1 LLM 冻结无冲突)
    shared_params_dict = None
    if stage == 2 and cfg.use_pcgrad:
        shared_params_dict = get_shared_params_dict(raw_model)
        if is_main(rank):
            logging.info(
                f"[PCGrad] PCGrad enabled on {len(shared_params_dict)} shared params "
                f"(every {cfg.pcgrad_interval} steps)"
            )

    if is_main(rank):
        logging.info(
            f"\n{'=' * 70}\n  Stage {stage} │ Epochs: {n_epochs} │ "
            f"Steps: {total_steps} │ Warmup: {warmup_steps}\n{'=' * 70}"
        )

    start_epoch = 0
    if resume_ckpt is not None and resume_ckpt.get("stage") == stage:
        start_epoch = int(resume_ckpt.get("epoch_completed", 0))
        if start_epoch >= n_epochs:
            if is_main(rank):
                logging.info(
                    f"Stage {stage} checkpoint already completed {start_epoch} "
                    "epoch(s); advancing to the next stage."
                )
            return global_step, best_miou

    for epoch in range(start_epoch, n_epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        tracker = MetricTracker([
            "loss_total", "loss_ce", "loss_mask", "loss_pose", "loss_prefix",
            "sigma_text", "sigma_mask", "sigma_pose", "beta",
        ])
        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            group_start = (step // cfg.grad_accum_steps) * cfg.grad_accum_steps
            accum_divisor = min(
                cfg.grad_accum_steps, len(train_loader) - group_start
            )
            loss_detail = train_one_step(
                model, batch, device, stage, global_step, cfg, log_vars,
                shared_params_dict=shared_params_dict,  # [PCGrad]
                total_steps=schedule_total_steps,       # CAPD α(t), paper Eq. (37)
                grad_accum_divisor=accum_divisor,
            )
            tracker.update(loss_detail, n=batch["input_ids"].shape[0])

            should_step = (
                (step + 1) % cfg.grad_accum_steps == 0
                or (step + 1) == len(train_loader)
            )
            if should_step:
                # log_vars are external leaf parameters; DDP cannot discover
                # them from model.forward, so synchronize their gradients here.
                if dist.is_initialized() and dist.get_world_size() > 1:
                    for value in log_vars.values():
                        # Every rank must enter every collective even when its
                        # local heterogeneous batch lacks this task.
                        if value.grad is None:
                            value.grad = torch.zeros_like(value)
                        dist.all_reduce(value.grad, op=dist.ReduceOp.AVG)
                nn.utils.clip_grad_norm_(all_trainable, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # [MEM] Periodic cache cleanup for 32GB VRAM
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Paper Eq. (9): β(t) increases continuously across both stages.
                if hasattr(raw_model, 'pose_head') and hasattr(raw_model.pose_head, 'log_beta'):
                    beta_target = compute_beta_target(
                        global_step, schedule_total_steps,
                        beta_min=getattr(cfg, 'spatial_beta', 2.0),
                        beta_max=6.0,
                    )
                    # Monotonic guard: β must never decrease
                    current_beta = torch.exp(raw_model.pose_head.log_beta).item()
                    beta_target = max(beta_target, current_beta)

                    log_beta_target = math.log(max(beta_target, 1e-3))
                    with torch.no_grad():
                        raw_model.pose_head.log_beta.data.lerp_(
                            torch.tensor(log_beta_target, device=device), 0.01
                        )

                if is_main(rank) and global_step % cfg.log_every_n_steps == 0:
                    avgs = tracker.averages()
                    lr = scheduler.get_last_lr()[0]
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    logging.info(
                        f"[S{stage}|E{epoch + 1}/{n_epochs}|Step {global_step}] "
                        f"loss={avgs.get('loss_total', 0):.4f} "
                        f"ce={avgs.get('loss_ce', 0):.4f} "
                        f"mask={avgs.get('loss_mask', 0):.4f} "
                        f"pose={avgs.get('loss_pose', 0):.4f} "
                        f"pfx={avgs.get('loss_prefix', 0):.4f} "
                        f"cons={avgs.get('loss_consist', 0):.4f} "
                        f"vp={avgs.get('loss_vp', 0):.4f} "
                        f"vAcc={avgs.get('vp_acc', 0):.2f} "
                        f"α={avgs.get('capd_alpha', 0):.2f} "
                        f"g={avgs.get('vp_gate', 0):.2f} "
                        f"σ_t={avgs.get('sigma_text', 0):.3f} "
                        f"σ_m={avgs.get('sigma_mask', 0):.3f} "
                        f"σ_p={avgs.get('sigma_pose', 0):.3f} "
                        f"β={avgs.get('beta', 0):.2f} "
                        f"lr={lr:.2e} VRAM={mem:.1f}GB"
                    )
                    if writer:
                        for k, v in avgs.items():
                            writer.add_scalar(f"train_s{stage}/{k}", v, global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                        writer.add_scalar("train/vram_gb", mem, global_step)
                    tracker.reset()

                if global_step % cfg.eval_every_n_steps == 0:
                    barrier()
                    # [MEM] Clear cache before eval
                    torch.cuda.empty_cache()

                    train_metrics = evaluate(
                        model, train_loader, device, cfg, stage, global_step, log_vars,
                        split_name="Train", max_batches=40,
                    )
                    val_metrics = evaluate(
                        model, val_loader, device, cfg, stage, global_step, log_vars,
                        # Full validation is required: records are grouped by
                        # AG/PR/VQA, so a prefix-only evaluation is biased.
                        split_name="Val", max_batches=None,
                    )
                    if is_main(rank):
                        for split_name, metrics in [("Train", train_metrics), ("Val", val_metrics)]:
                            logging.info(
                                f"  [Eval @ {global_step}] {split_name}: "
                                f"loss={metrics.get('loss_total', 0):.4f} "
                                f"ce={metrics.get('loss_ce', 0):.4f} "
                                f"cIoU(otsu)={metrics.get('ciou', 0):.4f} "
                                f"cIoU(0.5)={metrics.get('ciou_fixed', 0):.4f} "
                                f"gIoU={metrics.get('giou', 0):.4f} "
                                f"mIoU={metrics.get('miou', 0):.4f} "
                                f"softIoU={metrics.get('soft_iou', 0):.4f} "
                                f"trans={metrics.get('translation_error_px', 0):.2f}px "
                                f"angle={metrics.get('angle_error_deg', 0):.3f}° "
                                f"width={metrics.get('width_error_px', 0):.2f}px"
                            )
                            if writer:
                                prefix = f"{split_name.lower()}_s{stage}"
                                for k, v in metrics.items():
                                    if isinstance(v, (int, float)):
                                        writer.add_scalar(f"{prefix}/{k}", v, global_step)

                        # Select the best checkpoint only on the validation split.
                        cur_ciou = val_metrics.get("ciou", 0.0)
                        is_best = cur_ciou > best_miou
                        if is_best:
                            best_miou = cur_ciou
                            save_checkpoint(
                                state={
                                    "global_step": global_step, "stage": stage,
                                    "epoch": epoch,
                                    "model": raw_model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "best_miou": best_miou,
                                    "log_vars": {k: v.data.clone() for k, v in log_vars.items()},
                                },
                                save_dir=cfg.output_dir,
                                filename="best_model.pth",
                                is_best=False,  # 文件名直接就是 best_model.pth，不需要再复制一份
                            )
                            logging.info(
                                f"  ★ New best model saved at step {global_step} "
                                f"(stage {stage}, cIoU={cur_ciou:.4f})"
                            )
                    barrier()

        elapsed = time.time() - t_epoch
        if is_main(rank):
            logging.info(f"  Epoch {epoch + 1}/{n_epochs} done in {elapsed / 60:.1f} min")
            save_checkpoint(
                state={
                    "global_step": global_step,
                    "stage": stage,
                    "epoch": epoch,
                    "epoch_completed": epoch + 1,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "log_vars": {k: v.data.clone() for k, v in log_vars.items()},
                },
                save_dir=cfg.output_dir,
                filename="last_model.pth",
                is_best=False,
            )
            logging.info("  Recovery checkpoint updated: last_model.pth")
        barrier()

    return global_step, best_miou


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Paper-aligned TRP-Grasp training")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--llm_name_or_path", type=str, default=None)
    p.add_argument("--per_device_batch_size", type=int, default=None)
    p.add_argument("--stage1_epochs", type=int, default=None)
    p.add_argument("--stage2_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--use_4bit_quant", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    cfg = VLAConfig()

    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        unknown_keys = sorted(set(yaml_cfg) - set(VLAConfig.__dataclass_fields__))
        if unknown_keys:
            raise ValueError(
                "Unknown config keys (possible stale options or typos): "
                + ", ".join(unknown_keys)
            )
        for k, v in yaml_cfg.items():
            setattr(cfg, k, v)

    for attr in ["data_root", "output_dir", "llm_name_or_path",
                 "per_device_batch_size", "stage1_epochs", "stage2_epochs",
                 "learning_rate"]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)
    if args.use_4bit_quant:
        cfg.use_4bit_quant = True
    if args.resume:
        cfg.resume_from = args.resume

    set_global_seed(cfg.seed, rank)
    validate_training_paths(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    log_level = logging.INFO if is_main(rank) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(cfg.log_dir, f"train_rank{rank}.log"), encoding="utf-8",
            ),
        ],
    )

    if is_main(rank):
        logging.info("=== Paper-aligned TRP-Grasp training ===")
        logging.info(f"World size: {world_size} GPUs")
        logging.info(f"Config: {cfg}")
        logging.info(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        logging.info(f"Legacy PCGrad: {'ENABLED' if cfg.use_pcgrad else 'DISABLED'}")
        free_mem = torch.cuda.mem_get_info(device)
        logging.info(f"GPU {local_rank}: {free_mem[0]/1e9:.1f}GB free / {free_mem[1]/1e9:.1f}GB total")

    model = build_vla_model(cfg, device=device)

    # Move all params to GPU
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if buf.device.type == 'cpu':
            buf.data = buf.data.to(device)
    for name, module in model.named_modules():
        if hasattr(module, '_parameters'):
            for pname, p in module._parameters.items():
                if p is not None and p.device.type == 'cpu':
                    module._parameters[pname] = p.to(device)

    if is_main(rank):
        cpu_params = [n for n, p in model.named_parameters() if p.device.type == 'cpu']
        if cpu_params:
            logging.warning(f"Still on CPU: {cpu_params[:5]}...")
        else:
            logging.info("All parameters on GPU ✓")
        allocated = torch.cuda.memory_allocated(device) / 1e9
        logging.info(f"VRAM after model load: {allocated:.2f}GB")

    # Extract log_var from model for DDP compatibility
    log_var_text = model.log_var_text.data.clone().to(device).requires_grad_(True)
    log_var_mask = model.log_var_mask.data.clone().to(device).requires_grad_(True)
    log_var_pose = model.log_var_pose.data.clone().to(device).requires_grad_(True)
    del model.log_var_text, model.log_var_mask, model.log_var_pose
    model.register_buffer('log_var_text', torch.tensor(0.0, device=device))
    model.register_buffer('log_var_mask', torch.tensor(0.0, device=device))
    model.register_buffer('log_var_pose', torch.tensor(0.0, device=device))

    if dist.is_initialized() and world_size > 1:
        model_ddp = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
        raw_model = model_ddp.module
    else:
        model_ddp = model
        raw_model = model

    log_vars = {
        "text": log_var_text,
        "mask": log_var_mask,
        "pose": log_var_pose,
    }

    global_step = 0
    best_miou = 0.0
    start_stage = 1
    resume_ckpt = None

    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        resume_ckpt = load_checkpoint(cfg.resume_from, raw_model, strict=False)
        global_step = resume_ckpt.get("global_step", 0)
        best_miou = resume_ckpt.get("best_miou", 0.0)
        start_stage = resume_ckpt.get("stage", 1)
        if "log_vars" in resume_ckpt:
            for k in log_vars:
                if k in resume_ckpt["log_vars"]:
                    log_vars[k].data.copy_(resume_ckpt["log_vars"][k])
        if is_main(rank):
            logging.info(f"Resumed from {cfg.resume_from} at step {global_step}")

    # Reuse the exact tokenizer that was resized before PEFT wrapping.
    tokenizer = ensure_special_tokens(raw_model.tokenizer)

    train_loader, val_loader = build_dataloaders(
        cfg, tokenizer, rank=rank, world_size=world_size,
    )

    writer = None
    if is_main(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=cfg.log_dir)
        except ImportError:
            logging.warning("TensorBoard not available.")

    stages = [args.stage] if args.stage in (1, 2) else [1, 2]
    for stage in stages:
        if stage < start_stage:
            continue
        barrier()
        global_step, best_miou = run_stage(
            stage=stage, model=model_ddp, raw_model=raw_model,
            train_loader=train_loader, val_loader=val_loader,
            cfg=cfg, device=device, rank=rank, writer=writer,
            log_vars=log_vars,
            global_step=global_step, best_miou=best_miou,
            resume_ckpt=resume_ckpt,
        )
        resume_ckpt = None
        barrier()
        if is_main(rank):
            logging.info(f"Stage {stage} complete. Best cIoU={best_miou:.4f}")

    if is_main(rank):
        logging.info(f"Training complete. Best cIoU={best_miou:.4f}")
        logging.info(f"Best checkpoint saved at: {os.path.join(cfg.output_dir, 'best_model.pth')}")
        logging.info(
            "The full multimodal recovery state is stored in last_model.pth; "
            "Qwen2 LoRA is intentionally not merged separately."
        )
        if writer:
            writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
