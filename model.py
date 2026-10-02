"""TRP-Grasp model aligned with the paper architecture.

The model combines frozen DINOv2/SigLIP encoders, VA-Fusion, a Perceiver
adapter and visual prefixes with Qwen2-LoRA.  A shared semantic-spatial
decoder feeds the mask branch and the CAPD pose branch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
#  两个原创模块 (CAPD + VA-Fusion) 内联在本文件, 无需额外 import 外部文件
#   - CAPDPoseHead: Consistency-Aligned Pose Decoding (主创新)
#   - VAFusion:     Viewpoint-Aware Cross-Modal Fusion (副创新)
# 详见文件内 [CAPD] 和 [VA-Fusion] 标记的类定义
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 0. Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal_2d_pos_enc(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns [H*W, dim] 2-D sinusoidal positional encoding."""
    assert dim % 4 == 0
    half = dim // 2
    div = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / half)
    )
    row = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1) * div
    col = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1) * div
    row_enc = torch.cat([row.sin(), row.cos()], dim=-1)
    col_enc = torch.cat([col.sin(), col.cos()], dim=-1)
    row_enc = row_enc.unsqueeze(1).expand(H, W, half)
    col_enc = col_enc.unsqueeze(0).expand(H, W, half)
    return torch.cat([row_enc, col_enc], dim=-1).reshape(H * W, dim)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perceiver Resampler
# ─────────────────────────────────────────────────────────────────────────────

class PerceiverResampler(nn.Module):
    """
    Compresses [B, N_vit, d_vit] → [B, M, d_llm] via learned cross-attention
    queries with 2D positional encoding. Preserves spatial topology.
    """

    def __init__(
        self,
        vit_dim: int,
        llm_dim: int,
        num_queries: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.llm_dim = llm_dim

        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.kv_proj = nn.Linear(vit_dim, llm_dim, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "xattn": nn.MultiheadAttention(llm_dim, num_heads, batch_first=True, dropout=0.0),
                "ff": nn.Sequential(
                    nn.Linear(llm_dim, llm_dim * 4),
                    nn.GELU(),
                    nn.Linear(llm_dim * 4, llm_dim),
                ),
                "norm_q": nn.LayerNorm(llm_dim),
                "norm_kv": nn.LayerNorm(llm_dim),
                "norm_ff": nn.LayerNorm(llm_dim),
            }))
        self.out_norm = nn.LayerNorm(llm_dim)
        self._pos_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _get_pos_enc(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = _sinusoidal_2d_pos_enc(H, W, self.llm_dim, device)
        return self._pos_cache[key]

    def forward(self, vit_feats: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B = vit_feats.shape[0]
        H_p, W_p = hw

        kv = self.kv_proj(vit_feats)
        pos = self._get_pos_enc(H_p, W_p, kv.device)
        kv = kv + pos.unsqueeze(0)

        q = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            q_n = layer["norm_q"](q)
            kv_n = layer["norm_kv"](kv)
            attn_out, _ = layer["xattn"](q_n, kv_n, kv_n)
            q = q + attn_out
            q = q + layer["ff"](layer["norm_ff"](q))

        return self.out_norm(q)


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 [OPT-5] Visual Prefix Tuning
# ─────────────────────────────────────────────────────────────────────────────

class VisualPrefixTuning(nn.Module):
    """Paper Eq. (8): learnable visual prefixes prepended before Qwen2."""

    def __init__(self, llm_dim: int, num_prefix: int = 4):
        super().__init__()
        self.num_prefix = num_prefix
        self.prefix_tokens = nn.Parameter(torch.randn(1, num_prefix, llm_dim) * 0.02)
        # Kept for old-checkpoint shape compatibility; the paper-aligned
        # configs set prefix_contrastive_weight=0, so it adds no objective.
        self.cluster_proj = nn.Sequential(
            nn.Linear(llm_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, visual_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_tokens: [B, M, d_llm] from Perceiver Resampler
        Returns:
            prefixed_tokens: [B, num_prefix + M, d_llm]
            prefix_features: [B, num_prefix, 64] for contrastive loss
        """
        B = visual_tokens.shape[0]
        prefix = self.prefix_tokens.expand(B, -1, -1)
        prefixed = torch.cat([prefix, visual_tokens], dim=1)
        prefix_feat = self.cluster_proj(prefix)
        return prefixed, prefix_feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. [OPT-1,2,3] ASPP Module + PixelShuffle FPN
# ─────────────────────────────────────────────────────────────────────────────

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for large receptive field."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ))
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        # Fuse all branches
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        gap_out = self.gap(x)
        gap_out = F.interpolate(gap_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        outs.append(gap_out)
        return self.fuse(torch.cat(outs, dim=1))


class PixelShuffleUpsample(nn.Module):
    """Learnable 2x upsampling via PixelShuffle (sub-pixel convolution)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.shuffle(self.conv(x))))


class SharedFPNDecoder(nn.Module):
    """
    FPN with PixelShuffle upsampling and shallow feature injection.
    
    [OPT-2] PixelShuffle replaces bilinear interpolation
    [OPT-3] Accepts additional shallow features (pseudo-P2) for high-freq edges
    """

    def __init__(
        self,
        vit_dim: int = 1024,
        llm_dim: int = 3584,
        fpn_channels: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_shallow: int = 2,  # Number of shallow ViT layers to inject
    ):
        super().__init__()
        self.fpn_channels = fpn_channels
        self.num_levels = num_levels
        self.num_shallow = num_shallow

        # Lateral projections for deep ViT levels
        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Linear(vit_dim, fpn_channels), nn.LayerNorm(fpn_channels))
            for _ in range(num_levels)
        ])

        # [OPT-3] Lateral projections for shallow features
        self.shallow_laterals = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vit_dim, fpn_channels), nn.LayerNorm(fpn_channels),
                # Extra conv to refine shallow features
            )
            for _ in range(num_shallow)
        ])
        self.shallow_fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False),
                nn.GroupNorm(8, fpn_channels),
                nn.GELU(),
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, fpn_channels),
                nn.GELU(),
            )
            for _ in range(num_shallow)
        ])

        # Query projection from LLM hidden state
        self.query_proj = nn.Linear(llm_dim, fpn_channels)

        # Language-guided cross-attention at each level
        self.xattns = nn.ModuleList([
            nn.MultiheadAttention(fpn_channels, num_heads, batch_first=True, dropout=0.0)
            for _ in range(num_levels)
        ])
        self.xattn_norms_q = nn.ModuleList([nn.LayerNorm(fpn_channels) for _ in range(num_levels)])
        self.xattn_norms_kv = nn.ModuleList([nn.LayerNorm(fpn_channels) for _ in range(num_levels)])

        # Top-down refinement convolutions
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, fpn_channels),
                nn.GELU(),
            )
            for _ in range(num_levels - 1)
        ])

        # [OPT-2] PixelShuffle-based upsampling
        self.upsample_blocks = nn.ModuleList([
            PixelShuffleUpsample(fpn_channels)
            for _ in range(4)  # up to 4 rounds of 2x
        ])

        # Final refinement after shallow fusion
        self.final_refine = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, fpn_channels),
            nn.GELU(),
        )

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],  # num_levels × [B, N, d_vit]
        token_hidden: torch.Tensor,              # [B, d_llm]
        hw: Tuple[int, int],
        target_size: int = 336,
        shallow_feats: Optional[List[torch.Tensor]] = None,  # [OPT-3]
    ) -> torch.Tensor:
        """Returns [B, fpn_channels, target_size, target_size]."""
        B = token_hidden.shape[0]
        H_p, W_p = hw
        device = token_hidden.device

        query = self.query_proj(token_hidden).unsqueeze(1)  # [B, 1, C]

        # Build laterally-projected + language-guided feature maps
        fmaps = []
        for lvl in range(self.num_levels):
            feat = multi_level_feats[lvl].to(device)
            x = self.laterals[lvl](feat)

            q_n = self.xattn_norms_q[lvl](query)
            kv_n = self.xattn_norms_kv[lvl](x)
            delta, _ = self.xattns[lvl](q_n, kv_n, kv_n)
            x = x + delta.expand_as(x)

            fm = x.permute(0, 2, 1).reshape(B, self.fpn_channels, H_p, W_p)
            fmaps.append(fm)

        # Top-down pathway
        td = fmaps[-1]
        for i in range(self.num_levels - 2, -1, -1):
            td = F.interpolate(td, size=fmaps[i].shape[-2:], mode='bilinear', align_corners=False)
            td = td + fmaps[i]
            td = self.td_convs[self.num_levels - 2 - i](td)

        # [OPT-2] Progressive PixelShuffle upsample
        up_idx = 0
        while td.shape[-1] < target_size and up_idx < len(self.upsample_blocks):
            if td.shape[-1] * 2 <= target_size:
                td = self.upsample_blocks[up_idx](td)
            else:
                # Last step: use interpolation to exact target size
                td = F.interpolate(td, size=(target_size, target_size),
                                   mode='bilinear', align_corners=False)
            up_idx += 1

        if td.shape[-1] != target_size:
            td = F.interpolate(td, size=(target_size, target_size),
                               mode='bilinear', align_corners=False)

        # [OPT-3] Inject shallow features
        if shallow_feats is not None:
            for si, (sfeat, lat, fuse_conv) in enumerate(
                zip(shallow_feats, self.shallow_laterals, self.shallow_fuse_convs)
            ):
                sfeat = sfeat.to(device)
                s_proj = lat(sfeat)  # [B, N, C]
                s_fm = s_proj.permute(0, 2, 1).reshape(B, self.fpn_channels, H_p, W_p)
                s_fm = F.interpolate(s_fm, size=td.shape[-2:], mode='bilinear', align_corners=False)
                td = fuse_conv(torch.cat([td, s_fm], dim=1))

        td = self.final_refine(td)
        return td  # [B, fpn_channels, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 3. [OPT-1] Mask Decoder Head with ASPP
# ─────────────────────────────────────────────────────────────────────────────

class MaskHead(nn.Module):
    """ASPP-enhanced affordance heatmap decoder."""

    def __init__(self, fpn_channels: int = 256):
        super().__init__()
        self.aspp = ASPPModule(fpn_channels, 64, dilations=(1, 3, 5))
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, fpn_feat: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.aspp(fpn_feat))  # [B, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 4. [OPT-4] Spatial Softmax with Dynamic Temperature
# ─────────────────────────────────────────────────────────────────────────────

class SpatialSoftmaxPoseHead(nn.Module):
    """Non-CAPD ablation head with direct center-conditioned angle sampling."""

    def __init__(self, fpn_channels: int = 256, llm_dim: int = 3584, beta_init: float = 2.0):
        super().__init__()
        self.fpn_channels = fpn_channels
        # [OPT-4] Learnable temperature parameter
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

        # Grasp confidence heatmap with ASPP
        self.grasp_aspp = ASPPModule(fpn_channels, 64, dilations=(1, 3, 5))
        self.grasp_out = nn.Conv2d(64, 1, 1)

        # Dense angle field with ASPP for better coverage of elongated objects
        self.angle_aspp = ASPPModule(fpn_channels, 64, dilations=(1, 3, 5))
        self.angle_out = nn.Conv2d(64, 2, 1)

        # [Width] Width head 现在融合视觉特征 + LLM token
        # 输入: visual pooled (fpn_channels) + LLM hidden (llm_dim) → 1
        self.width_visual_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # [B, C, 4, 4] = [B, C*16]
            nn.Flatten(start_dim=1),
        )
        width_in_dim = fpn_channels * 16 + llm_dim   # visual_pool + llm_token
        self.width_head = nn.Sequential(
            nn.Linear(width_in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    @property
    def beta(self) -> torch.Tensor:
        # Clamp β to [1.0, 30.0] for stability
        return torch.clamp(torch.exp(self.log_beta), min=1.0, max=30.0)

    def forward(
        self,
        fpn_feat: torch.Tensor,            # [B, C, H, W]
        token_hidden: torch.Tensor,        # [B, d_llm]
        gt_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gt_centers: PR ground-truth grasp centers during training. ``None``
                at inference, where predicted centers are used.
        """
        B = fpn_feat.shape[0]
        device = fpn_feat.device

        # 1. Grasp heatmap + Spatial Softmax with learnable temperature
        grasp_feat = self.grasp_aspp(fpn_feat)
        grasp_logits = self.grasp_out(grasp_feat)  # [B, 1, H, W]
        H, W = grasp_logits.shape[-2:]

        beta = self.beta  # [OPT-4] dynamic
        flat = grasp_logits.reshape(B, -1)
        prob = torch.softmax(beta * flat, dim=-1)
        prob_2d = prob.reshape(B, H, W)

        grid_x = torch.linspace(0, 1, W, device=device, dtype=prob_2d.dtype)
        grid_y = torch.linspace(0, 1, H, device=device, dtype=prob_2d.dtype)

        x_hat = (prob_2d * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=(1, 2))
        y_hat = (prob_2d * grid_y.unsqueeze(-1).unsqueeze(0)).sum(dim=(1, 2))

        # 2. Dense angle field with ASPP
        angle_feat = self.angle_aspp(fpn_feat)
        angle_field = self.angle_out(angle_feat)  # [B, 2, H, W]
        angle_field = F.normalize(angle_field, p=2, dim=1)

        if self.training and gt_centers is not None:
            sample_x = (gt_centers[:, 0] * 2 - 1).reshape(B, 1, 1, 1)
            sample_y = (gt_centers[:, 1] * 2 - 1).reshape(B, 1, 1, 1)
        else:
            # 推理时 (或无 GT 时) 用预测中心
            sample_x = (x_hat * 2 - 1).reshape(B, 1, 1, 1)
            sample_y = (y_hat * 2 - 1).reshape(B, 1, 1, 1)

        sample_grid = torch.cat([sample_x, sample_y], dim=-1)
        sampled = F.grid_sample(
            angle_field, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border',
        )
        sin2t = sampled[:, 0, 0, 0]
        cos2t = sampled[:, 1, 0, 0]
        norm = torch.sqrt(sin2t ** 2 + cos2t ** 2 + 1e-8)
        sin2t = sin2t / norm
        cos2t = cos2t / norm

        # 3. [Width] Width 头融合视觉特征 + LLM token
        # ─────────────────────────────────────────────────
        # 原 earlier implementation: 仅 LLM token → width, 模型只能记忆"试管平均宽度", 类别外样本失败
        # visual width head: 拼接 FPN 特征池化 (含图像中物体的实际像素尺寸信息)
        visual_pooled = self.width_visual_pool(fpn_feat)  # [B, C*16]
        # token_hidden 可能在不同 device, 对齐
        token_h = token_hidden.to(visual_pooled.device).to(visual_pooled.dtype)
        width_in = torch.cat([visual_pooled, token_h], dim=-1)
        w_norm = torch.sigmoid(self.width_head(width_in).squeeze(-1))

        pred_pose = torch.stack([x_hat, y_hat, w_norm, sin2t, cos2t], dim=-1)

        return {
            "pred_pose": pred_pose,
            "grasp_heatmap": grasp_logits,   # [Heatmap] 用于 gaussian_heatmap_loss
            "beta": beta,
        }


# ─────────────────────────────────────────────────────────────────────────────
# [CAPD] Consistency-Aligned Pose Decoding Head  (主创新)
# ─────────────────────────────────────────────────────────────────────────────
class CAPDPoseHead(nn.Module):
    """
    Consistency-Aligned Pose Decoding (CAPD).

    解决 train-inference sampling discrepancy:
      (a) Scheduled Centroid Annealing  — 采样点从 GT 质心退火到预测质心
      (b) Local-Window Angle Sampling   — k×k 窗口 confidence 加权聚合角度
      (c) pred_center 输出供 Centroid Consistency Loss 使用

    与 SpatialSoftmaxPoseHead 接口兼容, forward 新增 global_step/total_steps,
    返回新增 'pred_center' 和 'alpha'.
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        llm_dim: int = 3584,
        beta_init: float = 2.0,
        window_size: int = 5,
        anneal_warmup_ratio: float = 0.1,
        anneal_end_ratio: float = 0.7,
        alpha_end: float = 0.9,
        use_confidence_weighting: bool = True,
        use_progressive_sampling: bool = True,
        fixed_alpha: float = 0.9,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels
        self.window_size = window_size
        self.anneal_warmup_ratio = anneal_warmup_ratio
        self.anneal_end_ratio = anneal_end_ratio
        self.alpha_end = alpha_end
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("CAPD window_size must be a positive odd integer")
        if not 0.0 <= fixed_alpha <= 1.0:
            raise ValueError("CAPD fixed_alpha must be in [0, 1]")
        self.use_confidence_weighting = use_confidence_weighting
        self.use_progressive_sampling = use_progressive_sampling
        self.fixed_alpha = fixed_alpha

        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

        self.grasp_aspp = ASPPModule(fpn_channels, 64, dilations=(1, 3, 5))
        self.grasp_out = nn.Conv2d(64, 1, 1)

        self.angle_aspp = ASPPModule(fpn_channels, 64, dilations=(1, 3, 5))
        self.angle_out = nn.Conv2d(64, 2, 1)

        # Width 头 (沿用 visual width head: 视觉池化 + LLM token)
        self.width_visual_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(start_dim=1),
        )
        width_in_dim = fpn_channels * 16 + llm_dim
        self.width_head = nn.Sequential(
            nn.Linear(width_in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    @property
    def beta(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_beta), min=1.0, max=30.0)

    def compute_alpha(self, global_step: int, total_steps: int) -> float:
        if not self.use_progressive_sampling:
            return self.fixed_alpha
        if total_steps <= 0:
            return 0.0
        t_warm = self.anneal_warmup_ratio * total_steps
        t_end = self.anneal_end_ratio * total_steps
        if global_step <= t_warm:
            return 0.0
        if global_step >= t_end:
            return self.alpha_end
        prog = (global_step - t_warm) / max(t_end - t_warm, 1.0)
        return float(prog * self.alpha_end)

    def _local_window_sample(self, angle_field, grasp_conf, center_norm):
        """(b) k×k 窗口 confidence 加权聚合角度场. 返回 [B,2]."""
        B, _, H, W = angle_field.shape
        k = self.window_size
        device, dtype = angle_field.device, angle_field.dtype

        half = (k - 1) / 2.0
        offs = torch.arange(k, device=device, dtype=dtype) - half
        dx = offs.view(1, 1, k) * (2.0 / max(W - 1, 1))
        dy = offs.view(1, k, 1) * (2.0 / max(H - 1, 1))

        cx = (center_norm[:, 0] * 2 - 1).view(B, 1, 1)
        cy = (center_norm[:, 1] * 2 - 1).view(B, 1, 1)
        grid_x = (cx + dx).expand(B, k, k)
        grid_y = (cy + dy).expand(B, k, k)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        sampled_angle = F.grid_sample(
            angle_field, grid, mode="bilinear",
            align_corners=True, padding_mode="border",
        )  # [B,2,k,k]
        if not self.use_confidence_weighting:
            return sampled_angle.mean(dim=(2, 3))
        sampled_conf = F.grid_sample(
            grasp_conf, grid, mode="bilinear",
            align_corners=True, padding_mode="border",
        )  # [B,1,k,k]
        w = torch.softmax(sampled_conf.reshape(B, -1), dim=-1).reshape(B, 1, k, k)
        return (sampled_angle * w).sum(dim=(2, 3))  # [B,2]

    def forward(
        self,
        fpn_feat: torch.Tensor,
        token_hidden: torch.Tensor,
        gt_centers: Optional[torch.Tensor] = None,
        global_step: int = 0,
        total_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        B = fpn_feat.shape[0]
        device = fpn_feat.device

        # 1. Grasp heatmap + spatial softmax → 预测中心
        grasp_feat = self.grasp_aspp(fpn_feat)
        grasp_logits = self.grasp_out(grasp_feat)
        H, W = grasp_logits.shape[-2:]

        beta = self.beta
        flat = grasp_logits.reshape(B, -1)
        prob = torch.softmax(beta * flat, dim=-1)
        prob_2d = prob.reshape(B, H, W)

        grid_x = torch.linspace(0, 1, W, device=device, dtype=prob_2d.dtype)
        grid_y = torch.linspace(0, 1, H, device=device, dtype=prob_2d.dtype)
        x_hat = (prob_2d * grid_x.view(1, 1, W)).sum(dim=(1, 2))
        y_hat = (prob_2d * grid_y.view(1, H, 1)).sum(dim=(1, 2))
        pred_center = torch.stack([x_hat, y_hat], dim=-1)

        # 2. Angle field
        angle_feat = self.angle_aspp(fpn_feat)
        angle_field = self.angle_out(angle_feat)
        angle_field = F.normalize(angle_field, p=2, dim=1)

        # 3. [CAPD-a] Scheduled Centroid Annealing
        if self.training and gt_centers is not None:
            alpha = self.compute_alpha(global_step, total_steps)
            c_sample = (1.0 - alpha) * gt_centers + alpha * pred_center.detach()
        else:
            alpha = 1.0
            c_sample = pred_center

        # 4. [CAPD-b] Local-Window Angle Sampling
        agg = self._local_window_sample(angle_field, grasp_logits, c_sample)
        sin2t, cos2t = agg[:, 0], agg[:, 1]
        norm = torch.sqrt(sin2t ** 2 + cos2t ** 2 + 1e-8)
        sin2t = sin2t / norm
        cos2t = cos2t / norm

        # 5. Width 头 (visual width head)
        visual_pooled = self.width_visual_pool(fpn_feat)
        token_h = token_hidden.to(visual_pooled.device).to(visual_pooled.dtype)
        width_in = torch.cat([visual_pooled, token_h], dim=-1)
        w_norm = torch.sigmoid(self.width_head(width_in).squeeze(-1))

        pred_pose = torch.stack([x_hat, y_hat, w_norm, sin2t, cos2t], dim=-1)

        return {
            "pred_pose": pred_pose,
            "grasp_heatmap": grasp_logits,
            "pred_center": pred_center,        # [CAPD-c] 供 consistency loss
            "beta": beta,
            "alpha": torch.tensor(float(alpha), device=device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# [VA-Fusion] Viewpoint-Aware Cross-Modal Fusion  (副创新)
# ─────────────────────────────────────────────────────────────────────────────
class VAFusion(nn.Module):
    """Paper Eq. (12)-(17): sample-level VA-Fusion scalar gate.

    A coarse observation label constrains ``vp_logit`` only during training;
    inference derives the gate exclusively from the two visual streams.
    """

    def __init__(self, dinov2_dim=1024, siglip_dim=1152, fused_dim=1024, vp_embed_dim=64, use_viewpoint=True, fixed_gate=None):
        super().__init__()
        self.use_viewpoint = use_viewpoint  # [ABLATION]
        if fixed_gate is not None and not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("VA-Fusion fixed_gate must be in [0, 1]")
        self.fixed_gate = fixed_gate
        self.fused_dim = fused_dim
        self.vp_head = nn.Sequential(
            nn.Linear(dinov2_dim + siglip_dim, vp_embed_dim),
            nn.GELU(),
            nn.LayerNorm(vp_embed_dim),
        )
        self.vp_logit = nn.Linear(vp_embed_dim, 1)
        self.fusion_proj = nn.Sequential(
            nn.Linear(dinov2_dim + siglip_dim, fused_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, f_dino, f_siglip):
        B, N, _ = f_dino.shape
        g_dino = f_dino.mean(dim=1)
        g_siglip = f_siglip.mean(dim=1)
        global_feat = torch.cat([g_dino, g_siglip], dim=-1)

        vp_embed = self.vp_head(global_feat)
        vp_logit = self.vp_logit(vp_embed)
        if self.use_viewpoint and self.fixed_gate is not None:
            gate = torch.full((B, 1), self.fixed_gate, device=f_dino.device, dtype=f_dino.dtype)
        elif self.use_viewpoint:                    # [ABLATION] VA-Fusion on
            gate = torch.sigmoid(vp_logit)
        else:                                       # [ABLATION] baseline: fixed equal-weight
            gate = torch.full((B, 1), 0.5, device=f_dino.device, dtype=f_dino.dtype)

        g_expand = gate.view(B, 1, 1)
        f_dino_mod = g_expand * f_dino
        f_siglip_mod = (1.0 - g_expand) * f_siglip
        concat = torch.cat([f_dino_mod, f_siglip_mod], dim=-1)
        fused = self.fusion_proj(concat)

        return fused, gate, vp_logit


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dual Vision Encoder with shallow feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class DualVisionEncoder(nn.Module):
    """
    DINOv2-L + SigLIP-SO400M dual-stream encoder.
    
    [OPT-3] Now extracts shallow layers (1, 2) as pseudo-P2 features.
    """

    def __init__(
        self,
        dinov2_model,
        siglip_model,
        dinov2_dim: int = 1024,
        siglip_dim: int = 1152,
        fused_dim: int = 1024,
        use_vaf: bool = True,
        vaf_fixed_gate: Optional[float] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2_model
        self.siglip = siglip_model
        self.fused_dim = fused_dim

        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        for p in self.siglip.parameters():
            p.requires_grad_(False)

        # [VA-Fusion] Viewpoint-Aware Cross-Modal Fusion 替换原等权融合
        # 原: self.fusion = nn.Sequential(Linear+GELU+LayerNorm)
        # 新: 视角门控动态调制双流融合
        self.va_fusion = VAFusion(
            dinov2_dim=dinov2_dim,
            siglip_dim=siglip_dim,
            fused_dim=fused_dim,
            vp_embed_dim=64,
            use_viewpoint=use_vaf,        # [ABLATION]
            fixed_gate=vaf_fixed_gate,
        )

        # Intermediate-level projections (DINOv2-only, layers 3/6/9)
        self.level_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dinov2_dim, fused_dim, bias=False),
                nn.LayerNorm(fused_dim),
            )
            for _ in range(3)
        ])

    def _extract_intermediate(
        self,
        pixel_values: torch.Tensor,
        model,
        target_layers: List[int] = [3, 6, 9],
        shallow_layers: List[int] = [1, 2],  # [OPT-3]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Extract intermediate + last + shallow hidden states (CLS removed)."""
        out = model(pixel_values=pixel_values, output_hidden_states=True)
        all_hs = out.hidden_states
        intermediates = []
        for l in target_layers:
            if l < len(all_hs):
                intermediates.append(all_hs[l][:, 1:, :])
            else:
                intermediates.append(all_hs[-1][:, 1:, :])

        # [OPT-3] Extract shallow features
        shallow = []
        for l in shallow_layers:
            if l < len(all_hs):
                shallow.append(all_hs[l][:, 1:, :])
            else:
                shallow.append(all_hs[1][:, 1:, :])

        last = out.last_hidden_state[:, 1:, :]
        return intermediates, last, shallow

    def forward(
        self,
        pixel_values_dino: torch.Tensor,
        pixel_values_siglip: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Tuple[int, int], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            multi_level_feats: 4 × [B, N, fused_dim]
            last_fused: [B, N, fused_dim]
            hw: (H_patch, W_patch)
            shallow_feats: 2 × [B, N, dinov2_dim] - raw shallow features
            vp_logit: [B, 1]  观察条件 logit（仅训练时可接收弱监督）
            vp_gate:  [B, 1]  视角门控 (日志用)
        """
        with torch.no_grad():
            dino_intermediates, dino_last, shallow_feats = self._extract_intermediate(
                pixel_values_dino.to(
                    next(self.dinov2.parameters()).device
                ),
                self.dinov2,
                target_layers=[3, 6, 9],
                shallow_layers=[1, 2],
            )

        with torch.no_grad():
            siglip_out = self.siglip.vision_model(
                pixel_values=pixel_values_siglip.to(
                    next(self.siglip.parameters()).device
                ),
                output_hidden_states=True,
            )
            siglip_last = siglip_out.last_hidden_state
            if siglip_last.shape[1] > dino_last.shape[1] + 1:
                siglip_last = siglip_last[:, 1:, :]

        target_device = next(self.va_fusion.parameters()).device
        B, N_d, _ = dino_last.shape
        siglip_last = siglip_last.to(target_device)

        _, N_s, _ = siglip_last.shape
        if N_s != N_d:
            siglip_last = siglip_last.permute(0, 2, 1)
            siglip_last = F.interpolate(siglip_last, size=N_d, mode='linear', align_corners=False)
            siglip_last = siglip_last.permute(0, 2, 1)

        dino_last = dino_last.to(target_device)
        # [VA-Fusion] 视角条件化融合
        last_fused, vp_gate, vp_logit = self.va_fusion(dino_last, siglip_last)

        multi_level = []
        for lvl_feat, proj in zip(dino_intermediates, self.level_fusion):
            multi_level.append(proj(lvl_feat.to(target_device)))
        multi_level.append(last_fused)

        H_p = W_p = int(math.sqrt(N_d))
        return multi_level, last_fused, (H_p, W_p), shallow_feats, vp_logit, vp_gate

# ─────────────────────────────────────────────────────────────────────────────
# 6. VLA Model (full assembly)
# ─────────────────────────────────────────────────────────────────────────────

GRASP_MASK_TOKEN = "<GRASP_MASK>"
GRASP_POSE_TOKEN = "<GRASP_POSE>"
SPECIAL_TOKENS = [GRASP_MASK_TOKEN, GRASP_POSE_TOKEN]


class VLAModel(nn.Module):
    """TRP-Grasp multimodal model for task-conditioned grasping."""

    def __init__(
        self,
        vision_encoder: DualVisionEncoder,
        llm_model,
        tokenizer,
        llm_dim: int,
        vit_fused_dim: int,
        img_size: int = 336,
        num_queries: int = 64,
        fpn_channels: int = 256,
        spatial_beta: float = 2.0,  # [OPT-4] Lower initial β
        lora_cfg=None,
        num_visual_prefix: int = 4,  # [OPT-5]
        use_capd: bool = True,        # [ABLATION]
        use_vaf: bool = True,         # [ABLATION]
        use_shared_decoder: bool = True,  # [ABLATION]
        capd_use_confidence_weighting: bool = True,
        capd_use_progressive_sampling: bool = True,
        capd_fixed_alpha: float = 0.9,
        capd_window_size: int = 5,
        capd_anneal_warmup: float = 0.1,
        capd_anneal_end: float = 0.7,
        capd_alpha_end: float = 0.9,
    ):
        super().__init__()
        self.use_capd = use_capd      # [ABLATION]
        self.use_vaf = use_vaf        # [ABLATION]
        self.use_shared_decoder = use_shared_decoder  # [ABLATION]

        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.llm_dim = llm_dim
        self.vit_fused_dim = vit_fused_dim

        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        llm_model.resize_token_embeddings(len(tokenizer))

        self.mask_token_id = tokenizer.convert_tokens_to_ids(GRASP_MASK_TOKEN)
        self.pose_token_id = tokenizer.convert_tokens_to_ids(GRASP_POSE_TOKEN)

        if lora_cfg is not None:
            from peft import get_peft_model
            self.llm = get_peft_model(llm_model, lora_cfg)
        else:
            self.llm = llm_model

        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()

        # Perceiver Resampler
        self.adapter = PerceiverResampler(
            vit_dim=vit_fused_dim,
            llm_dim=llm_dim,
            num_queries=num_queries,
        )

        # [OPT-5] Visual prefix tuning
        self.visual_prefix = VisualPrefixTuning(llm_dim, num_prefix=num_visual_prefix)

        # [OPT-2,3] Shared FPN with PixelShuffle + shallow injection
        if use_shared_decoder:
            self.shared_fpn = SharedFPNDecoder(
                vit_dim=vit_fused_dim,
                llm_dim=llm_dim,
                fpn_channels=fpn_channels,
                num_levels=4,
                num_shallow=2,
            )
        else:
            self.mask_decoder = SharedFPNDecoder(
                vit_dim=vit_fused_dim,
                llm_dim=llm_dim,
                fpn_channels=fpn_channels,
                num_levels=4,
                num_shallow=2,
            )
            self.pose_decoder = SharedFPNDecoder(
                vit_dim=vit_fused_dim,
                llm_dim=llm_dim,
                fpn_channels=fpn_channels,
                num_levels=4,
                num_shallow=2,
            )

        # [OPT-1] ASPP-enhanced heads
        self.mask_head = MaskHead(fpn_channels)
        # [ABLATION] CAPD or baseline SpatialSoftmax pose head
        if use_capd:
            self.pose_head = CAPDPoseHead(
                fpn_channels=fpn_channels,
                llm_dim=llm_dim,
                beta_init=spatial_beta,
                window_size=capd_window_size,
                anneal_warmup_ratio=capd_anneal_warmup,
                anneal_end_ratio=capd_anneal_end,
                alpha_end=capd_alpha_end,
                use_confidence_weighting=capd_use_confidence_weighting,
                use_progressive_sampling=capd_use_progressive_sampling,
                fixed_alpha=capd_fixed_alpha,
            )
        else:
            self.pose_head = SpatialSoftmaxPoseHead(
                fpn_channels=fpn_channels,
                llm_dim=llm_dim,
                beta_init=spatial_beta,
            )

        # Learnable uncertainty weights
        self.log_var_text = nn.Parameter(torch.tensor(0.0))
        self.log_var_mask = nn.Parameter(torch.tensor(0.0))
        self.log_var_pose = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pixel_values_dino: torch.Tensor,
        pixel_values_siglip: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        gt_pose: Optional[torch.Tensor] = None,
        stage: int = 2,
        global_step: int = 0,        # [CAPD] 退火用
        total_steps: int = 1,        # [CAPD] 退火用
    ) -> Dict[str, torch.Tensor]:

        # ── Vision encoding (multi-level + shallow + viewpoint) ───────────
        multi_level_feats, last_fused, hw, shallow_feats, vp_logit, vp_gate = self.vision_encoder(
            pixel_values_dino, pixel_values_siglip
        )

        # ── Perceiver Resampler ───────────────────────────────────────────
        visual_tokens = self.adapter(last_fused, hw)

        # ── [OPT-5] Visual prefix tuning ─────────────────────────────────
        visual_tokens, prefix_features = self.visual_prefix(visual_tokens)

        # ── Build LLM input ───────────────────────────────────────────────
        B = input_ids.shape[0]
        llm_device = next(self.llm.parameters()).device
        input_ids = input_ids.to(llm_device)
        attention_mask = attention_mask.to(llm_device)
        visual_tokens = visual_tokens.to(llm_device)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        full_attn = torch.cat([
            torch.ones(B, visual_tokens.shape[1], device=llm_device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)

        if labels is not None:
            pad_labels = torch.full(
                (B, visual_tokens.shape[1]), -100,
                dtype=labels.dtype, device=llm_device,
            )
            full_labels = torch.cat([pad_labels, labels.to(llm_device)], dim=1)
        else:
            full_labels = None

        # ── LLM forward ──────────────────────────────────────────────────
        _last_hs = {}

        def _capture_last_hidden(module, input, output):
            if isinstance(output, tuple):
                _last_hs["val"] = output[0]
            else:
                _last_hs["val"] = output

        try:
            base = self.llm.base_model.model if hasattr(self.llm, 'base_model') else self.llm
            layers = base.model.layers
            hook = layers[-1].register_forward_hook(_capture_last_hidden)
        except Exception:
            hook = None

        llm_out = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            labels=full_labels,
            output_hidden_states=False,
            use_cache=False,
        )

        if hook is not None:
            hook.remove()

        results: Dict[str, torch.Tensor] = {}
        if full_labels is not None:
            results["loss_ce"] = llm_out.loss

        # [OPT-5] Prefix features for contrastive loss
        results["prefix_features"] = prefix_features

        # ── Extract special token hidden states ──────────────────────────
        if "val" in _last_hs:
            last_hidden = _last_hs["val"]
        else:
            llm_out2 = self.llm(
                inputs_embeds=full_embeds, attention_mask=full_attn,
                output_hidden_states=True, use_cache=False,
            )
            last_hidden = llm_out2.hidden_states[-1]
            del llm_out2

        N_vis = visual_tokens.shape[1]
        text_hidden = last_hidden[:, N_vis:, :]

        mask_tok_hs = self._extract_token_hidden(text_hidden, input_ids, self.mask_token_id)
        pose_tok_hs = self._extract_token_hidden(text_hidden, input_ids, self.pose_token_id)

        # ── Shared FPN + task-specific heads ─────────────────────────────
        if self.use_shared_decoder:
            decoder_device = next(self.shared_fpn.parameters()).device
        else:
            decoder_device = next(self.mask_decoder.parameters()).device

        if mask_tok_hs is not None:
            if self.use_shared_decoder:
                fpn_mask = self.shared_fpn(
                    multi_level_feats, mask_tok_hs.to(decoder_device), hw, self.img_size,
                    shallow_feats=shallow_feats,  # [OPT-3]
                )
            else:
                fpn_mask = self.mask_decoder(
                    multi_level_feats, mask_tok_hs.to(decoder_device), hw, self.img_size,
                    shallow_feats=shallow_feats,  # [OPT-3]
                )
            heatmap = self.mask_head(fpn_mask)
            results["heatmap_logits"] = heatmap

        if pose_tok_hs is not None:
            if self.use_shared_decoder:
                fpn_pose = self.shared_fpn(
                    multi_level_feats, pose_tok_hs.to(decoder_device), hw, self.img_size,
                    shallow_feats=shallow_feats,  # [OPT-3]
                )
            else:
                decoder_device = next(self.pose_decoder.parameters()).device
                fpn_pose = self.pose_decoder(
                    multi_level_feats, pose_tok_hs.to(decoder_device), hw, self.img_size,
                    shallow_feats=shallow_feats,  # [OPT-3]
                )
            
            # 论文式(23)/(28): PR 记录以真值位姿中心 c* 作为 CAPD 训练采样条件。
            # PR 按异构监督定义没有 Mask，不能从全零 gt_mask 计算中心。
            gt_centers = None
            if self.training and gt_pose is not None:
                gt_centers = gt_pose[:, :2].to(decoder_device)
            
            if self.use_capd:                              # [ABLATION] CAPD head
                pose_out = self.pose_head(
                    fpn_pose, pose_tok_hs.to(decoder_device),
                    gt_centers=gt_centers,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            else:                                          # [ABLATION] baseline head
                pose_out = self.pose_head(
                    fpn_pose, pose_tok_hs.to(decoder_device),
                    gt_centers=gt_centers,
                )
            results["pred_pose"] = pose_out["pred_pose"]
            results["grasp_heatmap"] = pose_out["grasp_heatmap"]
            results["beta"] = pose_out["beta"]
            if self.use_capd and "pred_center" in pose_out:   # [ABLATION] CAPD-only
                results["pred_center"] = pose_out["pred_center"]
                results["capd_alpha"] = pose_out["alpha"]

        # VA-Fusion logit is consumed only by the optional coarse-label BCE.
        results["vp_logit"] = vp_logit
        results["vp_gate"] = vp_gate

        return results

    @staticmethod
    def _extract_token_hidden(
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
    ) -> Optional[torch.Tensor]:
        positions = (input_ids == token_id)
        if not positions.any():
            return None
        B, L, D = hidden.shape
        result = []
        for b in range(B):
            idx = positions[b].nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                result.append(hidden[b, idx[-1]])
            else:
                result.append(hidden[b, -1])
        return torch.stack(result, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_config(rank: int = 16, alpha: int = 32):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )


def build_vla_model(cfg, device: torch.device = None) -> VLAModel:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = None
    if getattr(cfg, "use_4bit_quant", True):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm_name_or_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    llm_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if bnb_cfg is not None:
        llm_kwargs["quantization_config"] = bnb_cfg
        llm_kwargs["device_map"] = {"": device or "cuda:0"}
    else:
        llm_kwargs["device_map"] = {"": device or "cuda:0"}

    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name_or_path, **llm_kwargs)
    old_vocab_size = llm.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != old_vocab_size:
        llm.resize_token_embeddings(len(tokenizer))
        # The special-token rows remain frozen with the Qwen2 backbone. Give
        # them stable semantic initializations instead of random new rows.
        with torch.no_grad():
            embedding = llm.get_input_embeddings().weight
            output_embedding = llm.get_output_embeddings().weight
            for special, seed_word in (
                (GRASP_MASK_TOKEN, "mask"),
                (GRASP_POSE_TOKEN, "pose"),
            ):
                special_id = tokenizer.convert_tokens_to_ids(special)
                seed_ids = tokenizer.encode(seed_word, add_special_tokens=False)
                seed_ids = [idx for idx in seed_ids if idx < old_vocab_size]
                if seed_ids:
                    source = embedding[seed_ids].float().mean(dim=0)
                    embedding[special_id].copy_(source.to(embedding.dtype))
                    if output_embedding.data_ptr() != embedding.data_ptr():
                        output_embedding[special_id].copy_(
                            output_embedding[seed_ids].float().mean(dim=0).to(
                                output_embedding.dtype
                            )
                        )

    if bnb_cfg is not None:
        from peft import prepare_model_for_kbit_training
        llm = prepare_model_for_kbit_training(
            llm, use_gradient_checkpointing=True,
        )

    dinov2 = AutoModel.from_pretrained(cfg.dinov2_name, torch_dtype=torch.bfloat16)
    siglip = AutoModel.from_pretrained(cfg.siglip_name, torch_dtype=torch.bfloat16)

    vision_enc = DualVisionEncoder(
        dinov2_model=dinov2,
        siglip_model=siglip,
        dinov2_dim=cfg.dinov2_dim,
        siglip_dim=cfg.siglip_dim,
        fused_dim=cfg.vit_fused_dim,
        use_vaf=getattr(cfg, "use_vaf", True),     # [ABLATION]
        vaf_fixed_gate=getattr(cfg, "vaf_fixed_gate", None),
    )

    lora_cfg = build_lora_config(cfg.lora_rank, cfg.lora_alpha)

    model = VLAModel(
        vision_encoder=vision_enc,
        llm_model=llm,
        tokenizer=tokenizer,
        llm_dim=cfg.llm_dim,
        vit_fused_dim=cfg.vit_fused_dim,
        img_size=cfg.img_size,
        num_queries=getattr(cfg, "num_queries", 64),
        fpn_channels=getattr(cfg, "fpn_channels", 256),
        spatial_beta=getattr(cfg, "spatial_beta", 2.0),  # [OPT-4] start low
        lora_cfg=lora_cfg,
        num_visual_prefix=getattr(cfg, "num_visual_prefix", 4),
        use_capd=getattr(cfg, "use_capd", True),   # [ABLATION]
        use_vaf=getattr(cfg, "use_vaf", True),     # [ABLATION]
        use_shared_decoder=getattr(cfg, "use_shared_decoder", True),  # [ABLATION]
        capd_use_confidence_weighting=getattr(cfg, "capd_use_confidence_weighting", True),
        capd_use_progressive_sampling=getattr(cfg, "capd_use_progressive_sampling", True),
        capd_fixed_alpha=getattr(cfg, "capd_fixed_alpha", 0.9),
        capd_window_size=getattr(cfg, "capd_window_size", 5),
        capd_anneal_warmup=getattr(cfg, "capd_anneal_warmup", 0.1),
        capd_anneal_end=getattr(cfg, "capd_anneal_end", 0.7),
        capd_alpha_end=getattr(cfg, "capd_alpha_end", 0.9),
    )
    return model
