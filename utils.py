"""Losses, metrics, optimization and checkpoint helpers for TRP-Grasp.

The active training path implements the paper's area-balanced mask objective,
direct double-angle pose objective, uncertainty weighting, Gaussian center
response supervision and weak VA-Fusion condition loss. Legacy helpers are
retained only for old-checkpoint compatibility and are not enabled by the
paper-aligned configurations.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ─────────────────────────────────────────────────────────────────────────────
# 1. Focal Loss + Dynamic Dice + Soft-IoU + Scale-Aware Penalty
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """Returns [B] per-sample loss when reduction='none', scalar when 'mean'."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    per_pixel = alpha_t * (1 - p_t) ** gamma * bce
    per_sample = per_pixel.mean(dim=(1, 2, 3))  # [B]
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample  # [B]


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = 'none',
) -> torch.Tensor:
    """Returns [B] per-sample loss when reduction='none'."""
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    per_sample = 1.0 - (2 * inter + eps) / (union + eps)  # [B]
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def soft_iou_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = 'none',
) -> torch.Tensor:
    """
    [OPT-17] Soft-IoU loss: uses continuous probability values instead of
    hard binary thresholding. Returns [B] per-sample loss.
    """
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = (probs + targets - probs * targets).sum(dim=(1, 2, 3))
    per_sample = 1.0 - (inter + eps) / (union + eps)  # [B]
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def generalized_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = 'none',
) -> torch.Tensor:
    """
    [OPT-19] Generalized Dice Loss with class-frequency weighting.
    Returns [B] per-sample loss.
    """
    probs = torch.sigmoid(logits)
    B = probs.shape[0]

    probs_flat = probs.reshape(B, -1)
    targets_flat = targets.reshape(B, -1)

    w_fg = 1.0 / (targets_flat.sum(dim=1) ** 2 + eps)      # [B]
    w_bg = 1.0 / ((1 - targets_flat).sum(dim=1) ** 2 + eps) # [B]

    inter_fg = (probs_flat * targets_flat).sum(dim=1)
    denom_fg = (probs_flat + targets_flat).sum(dim=1)
    inter_bg = ((1 - probs_flat) * (1 - targets_flat)).sum(dim=1)
    denom_bg = ((1 - probs_flat) + (1 - targets_flat)).sum(dim=1)

    per_sample = 1.0 - 2.0 * (
        w_fg * inter_fg + w_bg * inter_bg + eps
    ) / (
        w_fg * denom_fg + w_bg * denom_bg + eps
    )  # [B]
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def center_distance_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'none',
) -> torch.Tensor:
    """
    [OPT-11] Scale-aware distance penalty. Returns [B] per-sample penalty.
    """
    probs = torch.sigmoid(logits)
    B, _, H, W = probs.shape
    device = probs.device

    grid_y = torch.linspace(0, 1, H, device=device).reshape(1, 1, H, 1).expand(B, 1, H, W)
    grid_x = torch.linspace(0, 1, W, device=device).reshape(1, 1, 1, W).expand(B, 1, H, W)

    p_sum = probs.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    px = (probs * grid_x).sum(dim=(2, 3)) / p_sum.squeeze(-1).squeeze(-1)
    py = (probs * grid_y).sum(dim=(2, 3)) / p_sum.squeeze(-1).squeeze(-1)

    t_sum = targets.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    tx = (targets * grid_x).sum(dim=(2, 3)) / t_sum.squeeze(-1).squeeze(-1)
    ty = (targets * grid_y).sum(dim=(2, 3)) / t_sum.squeeze(-1).squeeze(-1)

    per_sample = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2 + 1e-8).squeeze(-1)  # [B]
    if reduction == 'mean':
        return per_sample.mean()
    return per_sample


def compute_scale_weight(
    targets: torch.Tensor,
    reference_area_ratio: float = 0.05,
    weight_min: float = 0.5,
    weight_max: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    [OPT-16] Scale Balanced Learning: compute per-sample inverse-frequency
    weights based on foreground area. Smaller objects get exponentially
    higher weight to combat large-object dominance.
    """
    B = targets.shape[0]
    areas = targets.reshape(B, -1).sum(dim=1)  # [B] pixel counts
    total_pixels = targets[0].numel()

    area_ratio = areas / max(total_pixels, 1)

    # Absolute reference is required because the paper uses batch/GPU=1.
    # Batch-wise mean normalization would otherwise force every weight to 1.
    weights = torch.sqrt(
        torch.as_tensor(reference_area_ratio, device=targets.device,
                        dtype=area_ratio.dtype)
        / area_ratio.clamp(min=eps)
    )
    weights = weights.clamp(min=weight_min, max=weight_max)

    return weights  # [B]


def mask_loss_dynamic(
    logits: torch.Tensor,
    targets: torch.Tensor,
    global_step: int,
    dice_ramp_steps: int = 4000,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    area_weight_min: float = 0.5,
    area_weight_max: float = 5.0,
    area_reference_ratio: float = 0.05,
) -> torch.Tensor:
    """Paper Eq. (27): area-balanced focal + ramped Dice/soft-IoU."""
    fl = focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma, reduction='none')    # [B]
    dl = dice_loss(logits, targets, reduction='none')          # [B]
    sil = soft_iou_loss(logits, targets, reduction='none')     # [B]

    # [OPT-8] Cosine annealing for Dice ramp
    if global_step >= dice_ramp_steps:
        w_dice = 1.0
    else:
        progress = global_step / max(1, dice_ramp_steps)
        w_dice = 0.5 * (1.0 - math.cos(math.pi * progress))

    # [OPT-16] Per-sample SBL weighting (safe: clamped to 0.5-5.0x)
    scale_w = compute_scale_weight(
        targets,
        reference_area_ratio=area_reference_ratio,
        weight_min=area_weight_min,
        weight_max=area_weight_max,
    )  # ω_i^M in Eq. (27)

    # [Mask] 不再加 center_distance_penalty
    per_sample_loss = (
        fl
        + w_dice * (0.6 * dl + 0.4 * sil)
    )  # [B]

    return (per_sample_loss * scale_w).mean()


# ─────────────────────────────────────────────────────────────────────────────
# [Heatmap] Gaussian Heatmap Loss for Grasp Center Supervision
# ─────────────────────────────────────────────────────────────────────────────
# 直接监督 SpatialSoftmaxPoseHead 的 grasp_logits，让其在 GT 中心形成高斯峰值
# 替代 earlier implementation 的 center_distance_penalty（全图期望，对小目标不稳定）
#
# 数学原理:
#   GT heatmap H_gt(i,j) = exp(-((i-cy)² + (j-cx)²) / (2σ²))
#   预测 P = sigmoid(grasp_logits)
#   L = MSE(P, H_gt)，并对前景区域加权 W_fg
# ─────────────────────────────────────────────────────────────────────────────

def build_gaussian_heatmap(
    centers: torch.Tensor,    # [B, 2] = (cx_norm, cy_norm) in [0, 1]
    H: int,
    W: int,
    sigma_px: float = 6.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成 GT 高斯热图供 heatmap 监督。
    
    Args:
        centers: 归一化中心坐标 [B, 2]
        H, W:    目标特征图尺寸
        sigma_px: 高斯标准差(像素)，建议: H*0.018 (336→6, 448→8)
    Returns:
        heatmap: [B, 1, H, W]，峰值=1.0
    """
    B = centers.shape[0]
    if device is None:
        device = centers.device
    
    # 生成像素坐标网格
    yy = torch.arange(H, device=device, dtype=torch.float32).reshape(1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).reshape(1, 1, W)
    
    # GT 中心(像素)
    cx = centers[:, 0:1].reshape(B, 1, 1) * (W - 1)  # [B, 1, 1]
    cy = centers[:, 1:2].reshape(B, 1, 1) * (H - 1)
    
    # 高斯
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2  # [B, H, W]
    heatmap = torch.exp(-dist_sq / (2.0 * sigma_px ** 2))  # [B, H, W]
    return heatmap.unsqueeze(1)  # [B, 1, H, W]


def gaussian_heatmap_loss(
    grasp_logits: torch.Tensor,    # [B, 1, H, W] 模型 raw logits
    gt_centers: torch.Tensor,      # [B, 2] = (cx_norm, cy_norm)
    sigma_px: float = 6.0,
    fg_weight: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    [Heatmap] 高斯热图监督损失。
    
    设计目标:
      1. 强迫 grasp_heatmap 在 GT 中心形成尖锐峰值
      2. 通过 fg_weight 让前景(高斯峰附近)梯度主导，不被背景稀释
      3. 与 mask_loss 分离，专门负责 grasp 中心定位
    
    数学:
      P = sigmoid(grasp_logits)
      H_gt = build_gaussian_heatmap(gt_centers)
      W = 1 + (fg_weight - 1) * H_gt   # 前景权重在峰值=fg_weight, 背景=1
      L = mean( W * (P - H_gt)² )
    
    Args:
        grasp_logits: 模型输出
        gt_centers: GT 中心归一化坐标 [0,1]
        sigma_px: 高斯 σ，与图像尺度匹配
        fg_weight: 前景区域权重倍数，建议 5-10
    """
    B, C, H, W = grasp_logits.shape
    assert C == 1, f"grasp_logits should be [B,1,H,W], got {grasp_logits.shape}"
    
    # 生成 GT 高斯
    H_gt = build_gaussian_heatmap(gt_centers, H, W, sigma_px=sigma_px,
                                   device=grasp_logits.device)
    
    # 预测概率
    P = torch.sigmoid(grasp_logits)
    
    # 加权 MSE
    weight_map = 1.0 + (fg_weight - 1.0) * H_gt  # 峰值 fg_weight, 背景 1.0
    sq_err = (P - H_gt) ** 2
    loss = (weight_map * sq_err).mean()
    return loss


def compute_centers_from_mask(gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    从 GT mask 提取归一化质心 (cx_norm, cy_norm)。
    
    用于:
      1. gaussian_heatmap_loss 的 gt_centers 输入
      2. PoseHead 训练时角度场的 GT 采样位置（解耦链式误差）
    
    Args:
        gt_mask: [B, 1, H, W] binary mask
    Returns:
        centers: [B, 2] = (cx_norm, cy_norm) in [0, 1]
    """
    B, _, H, W = gt_mask.shape
    device = gt_mask.device
    
    yy = torch.arange(H, device=device, dtype=torch.float32).reshape(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).reshape(1, 1, 1, W)
    
    mask_sum = gt_mask.sum(dim=(2, 3), keepdim=True).clamp(min=eps)  # [B,1,1,1]
    cx_px = (gt_mask * xx).sum(dim=(2, 3), keepdim=True) / mask_sum  # [B,1,1,1]
    cy_px = (gt_mask * yy).sum(dim=(2, 3), keepdim=True) / mask_sum
    
    cx_norm = (cx_px / max(W - 1, 1)).reshape(B)
    cy_norm = (cy_px / max(H - 1, 1)).reshape(B)
    centers = torch.stack([cx_norm, cy_norm], dim=-1)  # [B, 2]
    return centers


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 [OPT-15] PCGrad — Projected Conflicting Gradients
# ─────────────────────────────────────────────────────────────────────────────

def pcgrad_project(
    grads: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    [OPT-15] PCGrad: Project conflicting gradients onto each other's
    normal planes to eliminate destructive interference.

    For each pair of task gradients g_i, g_j:
      if g_i · g_j < 0 (conflict detected):
        g_i^PC = g_i - (g_i·g_j / ||g_j||^2) * g_j
    
    This removes the component of g_i that opposes g_j, ensuring
    the update improves the current task without harming others.
    """
    import random
    num_tasks = len(grads)
    projected = [g.clone() for g in grads]

    # Random task ordering for unbiased projection
    order = list(range(num_tasks))
    random.shuffle(order)

    for i in order:
        for j in order:
            if i == j:
                continue
            g_i = projected[i]
            g_j = grads[j]

            dot = (g_i * g_j).sum()
            if dot < 0:
                # Conflict detected: project g_i onto normal plane of g_j
                g_j_norm_sq = (g_j * g_j).sum().clamp(min=1e-12)
                projected[i] = g_i - (dot / g_j_norm_sq) * g_j

    return projected


def apply_pcgrad(
    model: nn.Module,
    losses: Dict[str, torch.Tensor],
    shared_params: List[nn.Parameter],
    scaler=None,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """
    [OPT-15] Apply PCGrad to resolve multi-task gradient conflicts.
    
    1. Compute per-task gradients for shared parameters
    2. Detect and resolve conflicts via normal-plane projection
    3. Apply the deconflicted gradient sum
    
    Returns loss values for logging.
    """
    task_grads = {}
    loss_values = {}

    for task_name, loss in losses.items():
        if loss is None or torch.isnan(loss):
            continue
        loss_values[task_name] = loss.item()

        # Compute gradient for this task only
        model.zero_grad()
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward(retain_graph=True)

        # Collect gradients from shared parameters
        task_grad = []
        for p in shared_params:
            if p.grad is not None:
                task_grad.append(p.grad.detach().clone().flatten())
            else:
                task_grad.append(torch.zeros(p.numel(), device=p.device))
        task_grads[task_name] = torch.cat(task_grad)

    if len(task_grads) < 2:
        return loss_values

    # Apply PCGrad
    grad_list = list(task_grads.values())
    projected = pcgrad_project(grad_list)

    # Sum projected gradients
    total_grad = sum(projected)

    # Apply back to shared parameters
    model.zero_grad()
    offset = 0
    for p in shared_params:
        numel = p.numel()
        if p.grad is None:
            p.grad = total_grad[offset:offset + numel].reshape(p.shape).clone()
        else:
            p.grad.copy_(total_grad[offset:offset + numel].reshape(p.shape))
        offset += numel

    return loss_values


# ─────────────────────────────────────────────────────────────────────────────
# [PCGrad] Memory-Efficient PCGrad on Shared Parameters Only (Post-Backward)
# ─────────────────────────────────────────────────────────────────────────────
# 与上面的 apply_pcgrad 不同: 那个需要 N 次 backward(retain_graph=True),
# 显存爆炸（N倍 graph 副本），不适用于 7B LLM + DDP + grad_accum=16 的设置。
#
# 本函数在 SINGLE backward 后, 通过 hook 提前缓存的 per-task 梯度做投影。
# 关键约束: 必须在 train_one_step 中通过 register_hook 收集任务梯度。
#
# 数学正确性 vs earlier implementation 伪 PCGrad:
#   earlier implementation: cosine_similarity(shared_flat[:min_len], mask_dir[:min_len])  ← 错误!
#       不同参数语义的前缀截断, 无意义
#   current implementation: 对每个共享参数 p, 收集 p.grad_from_mask, p.grad_from_pose, p.grad_from_text
#       三者维度严格相同, 做正确的逐参数投影
# ─────────────────────────────────────────────────────────────────────────────

def apply_pcgrad_post_hook(
    task_grad_buffers: Dict[str, Dict[str, torch.Tensor]],
    shared_params_dict: Dict[str, nn.Parameter],
) -> dict:
    """
    [PCGrad] 应用 PCGrad 到通过 hook 收集的任务梯度。
    
    Args:
        task_grad_buffers: {task_name: {param_name: per-task grad tensor}}
            必须由调用者提前通过 hook 填充
        shared_params_dict: {param_name: param} 共享参数字典
    
    Returns:
        info dict with conflict statistics
    """
    import random
    
    task_names = list(task_grad_buffers.keys())
    if len(task_names) < 2:
        return {"conflict_rate": 0.0, "n_tasks": len(task_names)}
    
    n_conflicts = 0
    n_total = 0
    
    # 对每个共享参数独立做 PCGrad
    for pname, p in shared_params_dict.items():
        # 收集所有任务对此参数的梯度
        per_param_grads = []
        for tname in task_names:
            g = task_grad_buffers[tname].get(pname, None)
            if g is not None:
                per_param_grads.append(g.flatten())
        
        if len(per_param_grads) < 2:
            continue
        
        # 随机顺序投影 (对称性)
        order = list(range(len(per_param_grads)))
        random.shuffle(order)
        projected = [g.clone() for g in per_param_grads]
        
        for i in order:
            for j in order:
                if i == j:
                    continue
                g_i = projected[i]
                g_j = per_param_grads[j]   # 用原始 g_j 做参考方向
                
                dot = (g_i * g_j).sum()
                if dot < 0:
                    n_conflicts += 1
                    g_j_norm_sq = (g_j * g_j).sum().clamp(min=1e-12)
                    projected[i] = g_i - (dot / g_j_norm_sq) * g_j
                n_total += 1
        
        # 求和投影后的梯度作为最终梯度
        new_grad = sum(projected).reshape(p.shape).to(p.dtype)
        if p.grad is None:
            p.grad = new_grad
        else:
            p.grad.copy_(new_grad)
    
    return {
        "conflict_rate": n_conflicts / max(n_total, 1),
        "n_tasks": len(task_names),
        "n_shared_params": len(shared_params_dict),
    }


def collect_task_gradients_hook(
    loss: torch.Tensor,
    task_name: str,
    shared_params_dict: Dict[str, nn.Parameter],
    task_grad_buffers: Dict[str, Dict[str, torch.Tensor]],
    grad_accum_steps: int = 1,
) -> None:
    """
    [PCGrad] 通过 torch.autograd.grad 收集单任务对 shared params 的梯度。
    
    旧实现的问题（导致 DDP 报 "Encountered gradient which is undefined"）:
      1) 用 loss.backward(retain_graph=True) 收集梯度 →
         不仅污染 shared params 的 .grad，也累加到非 shared 参数 (LLM/heads) 上;
         调用 N 次 → 非 shared 参数的 .grad 被错误累加 N 倍。
      2) train.py 用 requires_grad_(False) trick 阻止 shared params 被覆盖 →
         DDP 在构建时已经为这些参数注册了 reduce hook, requires_grad 中途变化
         不会被 DDP 感知，导致它对这些参数发起 allreduce 但 grad 是 None → 报错。
    
    新实现:
      - torch.autograd.grad(loss, [shared_params], retain_graph=True) 只对
        shared params 求梯度，**不写入** p.grad, **不污染**非 shared 参数。
      - 调用方无需再做 requires_grad_(False) 的 hack, 与 DDP 完全兼容。
    
    使用模式:
        for tname, tloss in {"mask": L_mask, "pose": L_pose, "text": L_ce}.items():
            collect_task_gradients_hook(tloss, tname, shared_dict, buffers, accum)
        (total_loss / accum).backward()  # 正常 backward, DDP 正常 allreduce
        apply_pcgrad_post_hook(buffers, shared_dict)  # 投影并覆盖 shared.grad
        # 多卡时还需 dist.all_reduce(p.grad) 同步 PCGrad 后的 shared 梯度
    """
    if loss is None or (torch.is_tensor(loss) and torch.isnan(loss).any()):
        return
    
    shared_param_list = list(shared_params_dict.values())
    shared_pname_list = list(shared_params_dict.keys())
    if len(shared_param_list) == 0:
        return
    
    scaled = loss / grad_accum_steps
    
    # autograd.grad: 只对 shared_param_list 求梯度, 不写入 .grad, 不污染其它参数
    # retain_graph=True 让后续任务也能反向 (最后一次 total_loss.backward 才释放图)
    # allow_unused=True 容忍某些 shared 参数对该任务损失无贡献 (返回 None)
    try:
        grads = torch.autograd.grad(
            scaled,
            shared_param_list,
            retain_graph=True,
            allow_unused=True,
        )
    except RuntimeError as e:
        # 如果某任务损失完全断开 graph (例如全 batch 都没有该任务的样本),
        # autograd.grad 会抛错。直接跳过此任务。
        return
    
    # 缓存该任务对每个共享参数的梯度
    task_grad_buffers[task_name] = {}
    for pname, g in zip(shared_pname_list, grads):
        if g is not None:
            task_grad_buffers[task_name][pname] = g.detach().clone()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Uncertainty-weighted Multi-task Loss with TIGHTENED Clamping
# ─────────────────────────────────────────────────────────────────────────────

LOG_VAR_MIN      = -2.0
LOG_VAR_MAX      = 2.0
# [Uncertainty] LOG_VAR_MAX_MASK 从 0.0 解锁到 2.0
# 原 current implementation 设 0.0 让 σ_m 卡在上限 1.0, 在 Stage 2 (LLM LoRA 解冻) 时:
#   - mask loss 数值 (1.0+) 比 ce/pose 大很多
#   - 模型想让 σ_m 涨大来降权重, 但被卡顶 → mask 通道一直压制其他梯度
#   - 表现: cIoU 还能慢涨, 但 trans/angle 完全停滞
# 解锁后 σ_m 可以涨到 ~2.7, mask 通道权重自动平衡, pose 通道梯度释放
LOG_VAR_MAX_MASK = 2.0  # 解锁, 让 uncertainty 自由学习

def uncertainty_weighted_loss(
    loss_ce: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    loss_pose: Optional[torch.Tensor],
    log_var_text: torch.Tensor,
    log_var_mask: torch.Tensor,
    log_var_pose: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    [OPT-9/12/14] Asymmetric clamping: mask gets tighter upper bound.
    """
    terms = []
    details = {}
    used_vars = set()

    lv_text = log_var_text.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    lv_mask = log_var_mask.clamp(LOG_VAR_MIN, LOG_VAR_MAX_MASK)
    lv_pose = log_var_pose.clamp(LOG_VAR_MIN, LOG_VAR_MAX)

    if loss_ce is not None and not torch.isnan(loss_ce):
        w = torch.exp(-lv_text)
        terms.append(w * loss_ce.to(lv_text.device) + lv_text)
        details["loss_ce"] = loss_ce.item()
        details["sigma_text"] = torch.exp(0.5 * lv_text).item()
        used_vars.add("text")

    if loss_mask is not None and not torch.isnan(loss_mask):
        w = torch.exp(-lv_mask)
        terms.append(w * loss_mask.to(lv_mask.device) + lv_mask)
        details["loss_mask"] = loss_mask.item()
        details["sigma_mask"] = torch.exp(0.5 * lv_mask).item()
        used_vars.add("mask")

    if loss_pose is not None and not torch.isnan(loss_pose):
        w = torch.exp(-lv_pose)
        terms.append(w * loss_pose.to(lv_pose.device) + lv_pose)
        details["loss_pose"] = loss_pose.item()
        details["sigma_pose"] = torch.exp(0.5 * lv_pose).item()
        used_vars.add("pose")

    if "text" not in used_vars:
        terms.append(0.0 * log_var_text)
    if "mask" not in used_vars:
        terms.append(0.0 * log_var_mask)
    if "pose" not in used_vars:
        terms.append(0.0 * log_var_pose)

    total = sum(terms) if terms else torch.tensor(0.0, device=log_var_text.device, requires_grad=True)
    details["loss_total"] = total.item()
    return total.squeeze(), details


# ─────────────────────────────────────────────────────────────────────────────
# 2.5 [OPT-10] Contrastive Clustering Loss for Visual Prefix
# ─────────────────────────────────────────────────────────────────────────────

def prefix_contrastive_loss(
    prefix_features: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    B, N, D = prefix_features.shape
    feat = F.normalize(prefix_features, p=2, dim=-1)
    sim = torch.bmm(feat, feat.transpose(1, 2))
    mask = ~torch.eye(N, dtype=torch.bool, device=feat.device).unsqueeze(0)
    off_diag = sim[mask].reshape(B, -1)
    loss = torch.relu(off_diag + 0.5).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation Metrics with Otsu + mIoU
# ─────────────────────────────────────────────────────────────────────────────

def _otsu_threshold(logits_1d: torch.Tensor) -> float:
    """[OPT-7] Compute Otsu's optimal threshold."""
    probs = torch.sigmoid(logits_1d).detach().float().cpu().numpy().flatten()
    if len(probs) == 0:
        return 0.5

    hist, bin_edges = np.histogram(probs, bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.5

    sum_total = (hist * bin_centers).sum()
    sum_bg = 0.0
    w_bg = 0
    max_var = 0.0
    best_thresh = 0.5

    for i in range(256):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            best_thresh = bin_centers[i]

    return float(best_thresh)


@torch.no_grad()
def compute_miou(
    heatmap_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    use_otsu: bool = True,
) -> float:
    """cIoU with optional Otsu adaptive threshold."""
    if use_otsu:
        B = heatmap_logits.shape[0]
        ious = []
        for b in range(B):
            thresh = _otsu_threshold(heatmap_logits[b])
            pred = (torch.sigmoid(heatmap_logits[b:b+1]) > thresh).float()
            gt_b = gt_mask[b:b+1]
            if pred.shape != gt_b.shape:
                gt_b = F.interpolate(gt_b.float(), size=pred.shape[-2:], mode='nearest')
            inter = (pred * gt_b).sum()
            union = (pred + gt_b).clamp(max=1).sum()
            ious.append((inter + 1e-6) / (union + 1e-6))
        return torch.stack(ious).mean().item()
    else:
        pred = (torch.sigmoid(heatmap_logits) > threshold).float()
        if pred.shape != gt_mask.shape:
            gt_mask = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode='nearest')
        inter = (pred * gt_mask).sum(dim=(1, 2, 3))
        union = (pred + gt_mask).clamp(max=1).sum(dim=(1, 2, 3))
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean().item()


@torch.no_grad()
def compute_iou_components(
    heatmap_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    use_otsu: bool = True,
) -> Dict[str, float]:
    """
    [OPT-18] Returns cIoU + gIoU components + per-sample IoUs for mIoU.
    mIoU groups by object class at the caller level.
    """
    B = heatmap_logits.shape[0]
    total_inter = 0.0
    total_union = 0.0
    ious = []

    for b in range(B):
        if use_otsu:
            thresh = _otsu_threshold(heatmap_logits[b])
        else:
            thresh = threshold

        pred = (torch.sigmoid(heatmap_logits[b:b+1]) > thresh).float()
        gt_b = gt_mask[b:b+1]
        if pred.shape != gt_b.shape:
            gt_b = F.interpolate(gt_b.float(), size=pred.shape[-2:], mode='nearest')

        inter = (pred * gt_b).sum()
        union = (pred + gt_b).clamp(max=1).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
        total_inter += inter.item()
        total_union += union.item()

    # [OPT-18] Also compute Soft-IoU metric (continuous, no threshold)
    probs = torch.sigmoid(heatmap_logits)
    if probs.shape != gt_mask.shape:
        gt_mask_r = F.interpolate(gt_mask.float(), size=probs.shape[-2:], mode='nearest')
    else:
        gt_mask_r = gt_mask
    soft_inter = (probs * gt_mask_r).sum(dim=(1, 2, 3))
    soft_union = (probs + gt_mask_r - probs * gt_mask_r).sum(dim=(1, 2, 3))
    soft_iou = ((soft_inter + 1e-6) / (soft_union + 1e-6)).mean().item()

    return {
        "ciou": torch.stack(ious).mean().item(),
        "inter_sum": total_inter,
        "union_sum": total_union,
        "per_sample_ious": [iou.item() for iou in ious],
        "soft_iou": soft_iou,
    }


@torch.no_grad()
def compute_mean_iou(
    heatmap_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    use_otsu: bool = True,
) -> float:
    """
    [OPT-18] mIoU: Mean IoU across all samples in the batch.
    Unlike cIoU which is identical for single-class, mIoU here serves
    as a per-batch average that can be grouped by class at eval time.
    """
    B = heatmap_logits.shape[0]
    ious = []

    for b in range(B):
        if use_otsu:
            thresh = _otsu_threshold(heatmap_logits[b])
        else:
            thresh = threshold

        pred = (torch.sigmoid(heatmap_logits[b:b+1]) > thresh).float()
        gt_b = gt_mask[b:b+1]
        if pred.shape != gt_b.shape:
            gt_b = F.interpolate(gt_b.float(), size=pred.shape[-2:], mode='nearest')

        inter = (pred * gt_b).sum()
        union = (pred + gt_b).clamp(max=1).sum()
        iou = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou)

    if len(ious) == 0:
        return 0.0
    return torch.stack(ious).mean().item()


@torch.no_grad()
def compute_pose_error(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    img_size: int = 336,
) -> Dict[str, float]:
    pred_xy = pred_pose[:, :2] * img_size
    gt_xy = gt_pose[:, :2] * img_size
    trans_err = torch.norm(pred_xy - gt_xy, dim=-1).mean().item()

    pred_angle = torch.atan2(pred_pose[:, 3], pred_pose[:, 4]) * 0.5
    gt_angle = torch.atan2(gt_pose[:, 3], gt_pose[:, 4]) * 0.5
    angle_diff = torch.abs(pred_angle - gt_angle)
    angle_diff = torch.where(angle_diff > math.pi / 2, math.pi - angle_diff, angle_diff)
    angle_err = torch.rad2deg(angle_diff).mean().item()

    pred_w = pred_pose[:, 2] * img_size
    gt_w = gt_pose[:, 2] * img_size
    width_err = torch.abs(pred_w - gt_w).mean().item()

    return {
        "translation_error_px": trans_err,
        "angle_error_deg": angle_err,
        "width_error_px": width_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 [OPT-6] Eigenvalue Ratio Gating — HARDENED
# ─────────────────────────────────────────────────────────────────────────────

def compute_eigenvalue_ratio(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    B = mask.shape[0]
    device = mask.device
    ratios = torch.ones(B, device=device)

    for b in range(B):
        m = (mask[b, 0] > threshold).float()
        if m.sum() < 10:
            continue

        H, W = m.shape
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(gy, gx, indexing='ij')

        fg_mask = m > 0
        coords_y = yy[fg_mask]
        coords_x = xx[fg_mask]

        if len(coords_y) < 3:
            continue

        mean_y = coords_y.mean()
        mean_x = coords_x.mean()
        cy = coords_y - mean_y
        cx = coords_x - mean_x

        cov_xx = (cx * cx).mean()
        cov_yy = (cy * cy).mean()
        cov_xy = (cx * cy).mean()

        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        disc = torch.sqrt(torch.clamp(trace * trace / 4 - det, min=1e-8))

        lambda1 = trace / 2 + disc
        lambda2 = trace / 2 - disc
        lambda2 = torch.clamp(lambda2, min=1e-8)

        ratios[b] = lambda1 / lambda2

    return ratios


def eigenvalue_pose_mask(
    gt_mask: torch.Tensor,
    low_thresh: float = 0.85,
    high_thresh: float = 1.15,
) -> torch.Tensor:
    """
    [OPT-6] Returns per-sample mask (1=non-symmetric, 0=symmetric).
    Used by symmetry_aware_pose_loss to choose loss mode per sample.
    """
    ratios = compute_eigenvalue_ratio(gt_mask)
    inv_ratios = 1.0 / ratios.clamp(min=1e-8)
    is_symmetric = (ratios >= low_thresh) & (ratios <= high_thresh)
    is_symmetric = is_symmetric | ((inv_ratios >= low_thresh) & (inv_ratios <= high_thresh))
    gate = (~is_symmetric).float().detach()
    return gate


# ─────────────────────────────────────────────────────────────────────────────
# 3.55 [P2-FIX] Symmetry-Aware Min-Over-Modes Pose Loss
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the old "skip symmetric objects entirely" strategy.
# For symmetric objects (cover slips, microplates), instead of zeroing loss,
# we compute loss for ALL equivalent grasp modes and take the minimum.
#
# Mathematical basis (practical Bingham approximation):
#   - 180°-symmetric (rectangle): double-angle encoding already handles it
#   - 90°-symmetric (square, top-down plate): 4 equivalent grasps
#     In double-angle space: (sin2θ, cos2θ) and (-sin2θ, -cos2θ) are the 2 modes
#   - We compute loss for both modes and take min → network learns nearest mode
#
# This gives the network a valid gradient for ALL samples (no starvation),
# while respecting the physical symmetry of the object.
# ─────────────────────────────────────────────────────────────────────────────

def symmetry_aware_pose_loss(
    pred_pose: torch.Tensor,   # [B, 5] = [x, y, w, sin2θ, cos2θ]
    gt_pose: torch.Tensor,     # [B, 5]
    gt_mask: torch.Tensor,     # [B, 1, H, W]
    smooth_l1_beta: float = 0.1,
    eigenvalue_low: float = 0.85,
    eigenvalue_high: float = 1.15,
    use_von_mises: bool = True,   # [Angle] 默认启用 vonMises
) -> Optional[torch.Tensor]:
    """
    [Angle] 集成 vonMises 角度损失到对称感知 pose loss
    
    核心改动 (替代 earlier implementation 的 sin/cos smooth_l1):
    --------------------------------------
    earlier implementation 角度损失: smooth_l1(sin2θ_pred, sin2θ_gt) + smooth_l1(cos2θ_pred, cos2θ_gt)
                  ↓ 问题: sin/cos 各自做 L1, 损失景观平坦，预测均值 (0, 1) 即得
                          MAE≈0.5, 模型完全坍塌到此安全点 → 实测 angle_err 死锁 37.3°
    
    current implementation 角度损失: 1 - cos(2θ_pred - 2θ_gt)
              = 1 - (sin2θ_p · sin2θ_g + cos2θ_p · cos2θ_g)   ← 内积形式!
                  ↓ 优势: 在均值处梯度最大(导出非零方向力), 自然 180°对称,
                          不需要离散化, 是 vonMises 分布最大似然的等价形式
    
    对称性处理仍然兼容:
      - 非对称物体: 用 mode 1 (标准 vonMises)
      - 对称物体: min(mode 1, mode 2), mode 2 是 +90° 旋转的目标
    """
    if pred_pose.shape[0] == 0:
        return None

    B = pred_pose.shape[0]
    device = pred_pose.device

    # Detect symmetric vs non-symmetric
    sym_gate = eigenvalue_pose_mask(gt_mask, eigenvalue_low, eigenvalue_high)  # [B] 1=non-sym, 0=sym

    # ── Position + width loss (always standard) ──
    loss_xyw = F.smooth_l1_loss(
        pred_pose[:, :3], gt_pose[:, :3], beta=smooth_l1_beta, reduction='none'
    ).mean(dim=1)  # [B]

    # ── Angle loss ──
    pred_angle = pred_pose[:, 3:5]   # [B, 2] = (sin2θ, cos2θ)
    gt_angle = gt_pose[:, 3:5]

    if use_von_mises:
        # [Angle] vonMises 角度损失 (内积形式)
        # 注意: pred 和 gt 都已经在 forward / encode_pose 中归一化为单位向量
        # cos(Δ) = sin·sin + cos·cos
        cos_delta_mode1 = (pred_angle * gt_angle).sum(dim=-1)  # [B]
        # mode 2: GT 旋转 90° → (sin2(θ+90°), cos2(θ+90°)) = (-sin2θ, -cos2θ)
        cos_delta_mode2 = (pred_angle * (-gt_angle)).sum(dim=-1)  # [B]
        
        # 各自损失: 1 - cos(Δ), 范围 [0, 2]
        loss_angle_mode1 = 1.0 - cos_delta_mode1  # [B]
        loss_angle_mode2 = 1.0 - cos_delta_mode2  # [B]
    else:
        # earlier implementation fallback: smooth L1 (保留兼容性, 但默认不用)
        loss_angle_mode1 = F.smooth_l1_loss(
            pred_angle, gt_angle, beta=smooth_l1_beta, reduction='none'
        ).mean(dim=1)
        loss_angle_mode2 = F.smooth_l1_loss(
            pred_angle, -gt_angle, beta=smooth_l1_beta, reduction='none'
        ).mean(dim=1)

    # 非对称物体用 mode1; 对称物体取 min
    loss_angle_min = torch.min(loss_angle_mode1, loss_angle_mode2)
    loss_angle = sym_gate * loss_angle_mode1 + (1 - sym_gate) * loss_angle_min

    # 总损失
    total = loss_xyw + loss_angle  # [B]
    return total.mean()


def paper_pose_loss(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    smooth_l1_beta: float = 0.1,
) -> Optional[torch.Tensor]:
    """Paper Eq. (28)-(30): Smooth-L1(x,y,w) + double-angle vector loss.

    The (sin 2θ, cos 2θ) representation already encodes the 180-degree
    equivalence of a parallel gripper, so no additional shape-derived mode is
    introduced here.
    """
    if pred_pose.shape[0] == 0:
        return None
    loss_xyw = F.smooth_l1_loss(
        pred_pose[:, :3], gt_pose[:, :3],
        beta=smooth_l1_beta, reduction="none",
    ).mean(dim=1)
    pred_angle = F.normalize(pred_pose[:, 3:5], p=2, dim=-1)
    gt_angle = F.normalize(gt_pose[:, 3:5], p=2, dim=-1)
    loss_ang = 1.0 - (pred_angle * gt_angle).sum(dim=-1)
    return (loss_xyw + loss_ang).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3.6 [NEW] β Temperature Annealing Schedule
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_target(
    global_step: int,
    total_steps: int,
    beta_min: float = 2.0,
    beta_max: float = 10.0,
) -> float:
    """Paper Eq. (9): smoothly increase inverse temperature β(t)."""
    progress = min(1.0, max(0.0, global_step / max(1, total_steps)))
    # Cosine ramp: 0→1 smoothly
    return beta_min + (beta_max - beta_min) * 0.5 * (1.0 - math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# 4. LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optimizer builder
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg, stage: int = 1) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    lora_params = []
    log_var_params = []

    NO_DECAY = {"bias", "LayerNorm.weight", "layer_norm.weight", "GroupNorm.weight"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "log_var" in name:
            log_var_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        elif any(nd in name for nd in NO_DECAY):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay, "lr": cfg.learning_rate},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": cfg.learning_rate},
        {"params": lora_params, "weight_decay": 0.0, "lr": cfg.lora_lr},
        {"params": log_var_params, "weight_decay": 0.0, "lr": cfg.learning_rate},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95), eps=1e-8)
        print("[Optimizer] Using 8-bit AdamW")
    except ImportError:
        optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        print("[Optimizer] Using standard AdamW")

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(save_dir, "best_model.pth"))
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    strict: bool = False,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
        if missing:
            print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[Checkpoint] Optimizer restore failed: {e}")
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[Checkpoint] Scheduler restore failed: {e}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric tracking
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    def __init__(self, keys):
        self.meters = {k: AverageMeter(k) for k in keys}

    def update(self, d: Dict[str, float], n: int = 1):
        for k, v in d.items():
            if k in self.meters:
                self.meters[k].update(v, n)
            else:
                # Auto-add new keys
                self.meters[k] = AverageMeter(k)
                self.meters[k].update(v, n)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self):
        for m in self.meters.values():
            m.reset()


# ─────────────────────────────────────────────────────────────────────────────
# [VA-Fusion] Viewpoint self-supervised loss (VA-Fusion 副创新配套损失)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_aspect_ratio(gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    从 GT mask 计算主轴长宽比 r = λ_max / λ_min (二阶矩特征值之比).
    细长物体 (正视试管/烧杯) r 大; 规则/各向同性物体 (俯视枪头盒) r 小.

    Args:
        gt_mask: [B, H, W] 或 [B, 1, H, W] 二值 mask
    Returns:
        ratio: [B] 长宽比
    """
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)
    B, H, W = gt_mask.shape
    device = gt_mask.device
    dtype = torch.float32

    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

    m = (gt_mask > 0.5).to(dtype)
    total = m.sum(dim=(1, 2)).clamp(min=eps)

    cx = (m * xs).sum(dim=(1, 2)) / total
    cy = (m * ys).sum(dim=(1, 2)) / total

    dx = xs - cx.view(B, 1, 1)
    dy = ys - cy.view(B, 1, 1)
    mxx = (m * dx * dx).sum(dim=(1, 2)) / total
    myy = (m * dy * dy).sum(dim=(1, 2)) / total
    mxy = (m * dx * dy).sum(dim=(1, 2)) / total

    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = torch.sqrt(torch.clamp(tr * tr / 4 - det, min=0.0))
    lam_max = tr / 2 + disc
    lam_min = tr / 2 - disc
    ratio = lam_max / lam_min.clamp(min=eps)
    return ratio


def viewpoint_self_supervised_loss(
    vp_logit: torch.Tensor,
    gt_mask: torch.Tensor,
    aspect_threshold: float = 3.0,
) -> torch.Tensor:
    """
    L_vp = BCE(sigmoid(vp_logit), 1[r > τ])
    伪标签来自 GT mask 长宽比, 无需人工视角标注 (VA-Fusion 自监督信号).
    """
    with torch.no_grad():
        ratio = compute_mask_aspect_ratio(gt_mask)
        pseudo_label = (ratio > aspect_threshold).to(vp_logit.dtype)
    vp_logit = vp_logit.squeeze(-1).float()
    pseudo_label = pseudo_label.float()
    return F.binary_cross_entropy_with_logits(vp_logit, pseudo_label)


def viewpoint_weak_label_loss(
    vp_logit: torch.Tensor,      # [B, 1] 视角推断 logit
    view_labels: torch.Tensor,   # [B] 类别派生弱标签: 1=正视, 0=俯视, -1=未知
) -> Optional[torch.Tensor]:
    """Paper Eq. (15): BCE on samples with a valid coarse condition label.

    ``view_label == -1`` denotes ambiguous/mixed observations and is excluded.
    Labels are not consumed by the model forward path or by inference.
    """
    valid = view_labels >= 0
    if not valid.any():
        return None
    logit = vp_logit.squeeze(-1).float()[valid]
    target = view_labels.float()[valid]
    return F.binary_cross_entropy_with_logits(logit, target)
