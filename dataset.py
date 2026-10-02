"""Paper-aligned heterogeneous dataset loader for TRP-Grasp.

AG records supervise masks, PR records supervise poses, and VQA records carry
language supervision only.  Coarse viewpoint labels are returned solely for
the VA-Fusion auxiliary loss and are never injected into language prompts.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ─────────────────────────────────────────────────────────────────────────────
# VA-Fusion 粗粒度观察条件弱标签（仅用于辅助 BCE，不注入语言输入）
# ─────────────────────────────────────────────────────────────────────────────
# 对构建协议中观察条件明确的类别派生弱标签；混合视角类别返回 unknown。
# ─────────────────────────────────────────────────────────────────────────────

VIEW_FRONT_CLASSES = {
    "beaker", "test_tube", "centrifuge_tube", "reagent_bottle",
    "pipette", "glass_rod",
    # 中文别名兼容
    "烧杯", "试管", "离心管", "试剂瓶", "移液器", "玻璃棒",
    "Beaker", "Test tube", "Centrifuge tube", "Reagent bottle",
    "Pipette", "Glass rod",
}

VIEW_TOP_CLASSES = {
    "coverslip", "microwell_plate",
    "盖玻片", "微孔板",
    "Coverslip", "Microwell plate",
}

# LabGrasp-Deploy-Real 中枪头盒同时存在俯视和侧视图，故不在固定映射中：
# 它返回 unknown，不参与 VA-Fusion 弱标签 BCE，但仍参与三类主任务。


def get_viewpoint_tag(object_class: str) -> str:
    """
    根据 object_class 返回视角标签.
    
    Returns:
        "[Front-view]" / "[Top-view]" / ""  (英文标签便于 LLM tokenizer 处理)
    """
    if not object_class:
        return ""
    
    obj_lower = object_class.lower().replace(" ", "_").replace("-", "_")
    
    if obj_lower in VIEW_FRONT_CLASSES or object_class in VIEW_FRONT_CLASSES:
        return "[Front-view]"
    if obj_lower in VIEW_TOP_CLASSES or object_class in VIEW_TOP_CLASSES:
        return "[Top-view]"
    return ""  # 未知类别不强行加标签


def _chat_messages(conversations: List[Dict]) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversations:
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    return messages


def tokenize_chat_with_assistant_labels(
    conversations: List[Dict], tokenizer, max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize the dialogue and supervise only the target assistant sequence Y_i.

    Equation (26) in the paper applies autoregressive CE to the target reasoning
    sequence, not to the system prompt or human instruction. LabGrasp records use
    one final assistant turn, so everything before that turn is masked with -100.
    """
    messages = _chat_messages(conversations)
    assistant_positions = [i for i, msg in enumerate(messages)
                           if msg["role"] == "assistant"]
    if not assistant_positions:
        raise ValueError("LabGrasp record has no assistant target turn")
    last_assistant = assistant_positions[-1]
    if last_assistant != len(messages) - 1:
        raise ValueError("LabGrasp record must end with the assistant target turn")

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:last_assistant], tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prefix = [f"<|system|>\n{SYSTEM_PROMPT}\n"]
        for msg in messages[1:last_assistant]:
            tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            prefix.append(f"{tag}\n{msg['content']}\n")
        prefix.append("<|assistant|>\n")
        prompt_text = "".join(prefix)
        full_text = prompt_text + messages[last_assistant]["content"]

    enc = tokenizer(
        full_text, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    prompt_ids = tokenizer(
        prompt_text, max_length=max_length, padding=False,
        truncation=True, return_tensors="pt",
    )["input_ids"].squeeze(0)

    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:min(len(prompt_ids), len(labels))] = -100
    if not (labels != -100).any():
        raise ValueError(
            "Assistant target was fully truncated; increase max_seq_len or shorten the record"
        )
    return input_ids, attention_mask, labels


# ─────────────────────────────────────────────────────────────────────────────
# Image normalisation
# ─────────────────────────────────────────────────────────────────────────────

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def _preprocess_image(
    img_bgr: np.ndarray, size: int, mean: Tuple, std: Tuple,
) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    img_f = img_rs.astype(np.float32) / 255.0
    img_f = (img_f - np.array(mean)) / np.array(std)
    return torch.from_numpy(img_f).permute(2, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pose encoding  [CRITICAL-FIX] Added is_flipped parameter
# ─────────────────────────────────────────────────────────────────────────────

def encode_pose(
    pose_dict: Dict,
    img_w: int,
    img_h: int,
    is_flipped: bool = False,
) -> torch.Tensor:
    """Normalise raw pose → [x_n, y_n, w_n, sin(2θ), cos(2θ)].

    [CRITICAL-FIX] When is_flipped=True:
      - x_norm = 1.0 - x_norm  (mirror x coordinate)
      - angle = -angle  (negate angle for horizontal flip)
    """
    x = pose_dict["x"] / max(img_w, 1)
    y = pose_dict["y"] / max(img_h, 1)
    w = pose_dict.get("width", 0.0) / max(img_w, 1)
    angle_deg = pose_dict.get("angle", 0.0)

    if is_flipped:
        x = 1.0 - x
        angle_deg = -angle_deg

    theta = math.radians(angle_deg)
    sin2t = math.sin(2 * theta)
    cos2t = math.cos(2 * theta)
    mag = math.sqrt(sin2t ** 2 + cos2t ** 2) + 1e-8
    return torch.tensor([x, y, w, sin2t / mag, cos2t / mag], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────────────────────

GRASP_MASK_TOKEN = "<GRASP_MASK>"
GRASP_POSE_TOKEN = "<GRASP_POSE>"
SPECIAL_TOKENS = [GRASP_MASK_TOKEN, GRASP_POSE_TOKEN]


def ensure_special_tokens(tokenizer):
    try:
        existing = tokenizer.additional_special_tokens or []
    except AttributeError:
        vocab = tokenizer.get_vocab()
        existing = [t for t in SPECIAL_TOKENS if t in vocab]

    to_add = [t for t in SPECIAL_TOKENS if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer


def extract_and_clean_special_tokens(
    conversations: List[Dict],
) -> Tuple[List[Dict], Optional[str], Optional[Dict]]:
    mask_path = None
    pose_dict = None
    cleaned = []

    mask_re = re.compile(r"<GRASP_MASK>(.*?)</GRASP_MASK>", re.DOTALL)
    pose_re = re.compile(r"<GRASP_POSE>(.*?)</GRASP_POSE>", re.DOTALL)

    for turn in conversations:
        value = turn.get("value", "")

        m = mask_re.search(value)
        if m:
            content = m.group(1).strip()
            if content:
                mask_path = content
            value = mask_re.sub(GRASP_MASK_TOKEN, value)

        p = pose_re.search(value)
        if p:
            content = p.group(1).strip()
            if content:
                try:
                    pose_dict = json.loads(content)
                except Exception:
                    try:
                        kv = dict(
                            item.split(":")
                            for item in content.replace(" ", "").split(",")
                            if ":" in item
                        )
                        pose_dict = {k: float(v) for k, v in kv.items()}
                    except Exception:
                        pass
            value = pose_re.sub(GRASP_POSE_TOKEN, value)

        cleaned.append({**turn, "value": value})
    return cleaned, mask_path, pose_dict


# ─────────────────────────────────────────────────────────────────────────────
# Chat template
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a laboratory robotic grasping assistant. Given an image and a task "
    "instruction, produce the verified task-grounded response in the supplied record. "
    "Use <GRASP_MASK> only for task-relevant region records and <GRASP_POSE> only "
    "for planar pose records. Visual-question-answering records require no spatial token."
)


# ─────────────────────────────────────────────────────────────────────────────
# VLA Dataset  [CRITICAL-FIX] Augmentation spatial synchronization
# ─────────────────────────────────────────────────────────────────────────────

class VLADataset(Dataset):
    """Multi-type VLA dataset for VQA, affordance grounding, and pose regression."""

    def __init__(
        self,
        jsonl_paths: List[str],
        data_root: str,
        tokenizer,
        max_length: int = 1024,
        dino_size: int = 336,
        siglip_size: int = 384,
        mask_size: int = 336,
        augment: bool = True,
        exclude_classes: Optional[List[str]] = None,
        strict_data: bool = True,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dino_size = dino_size
        self.siglip_size = siglip_size
        self.mask_size = mask_size
        self.augment = augment
        self.strict_data = strict_data
        self.exclude_classes = {
            str(name).strip().lower().replace("_", " ")
            for name in (exclude_classes or []) if str(name).strip()
        }

        self.entries: List[Dict] = []
        excluded = 0
        for p in jsonl_paths:
            if not os.path.exists(p):
                if strict_data:
                    raise FileNotFoundError(f"Required JSONL not found: {p}")
                print(f"[VLADataset] Warning: {p} not found, skipping.")
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError as exc:
                            if strict_data:
                                raise ValueError(f"Invalid JSON at {p}:{line_no}: {exc}") from exc
                            continue
                        obj = str(entry.get("ground_truth", {}).get(
                            "object_class", "")).strip().lower().replace("_", " ")
                        if obj in self.exclude_classes:
                            excluded += 1
                            continue
                        self.entries.append(entry)

        # Do not random.shuffle here: DistributedSampler must see the same index
        # mapping on every rank. It supplies epoch-wise shuffling consistently.
        print(
            f"[VLADataset] Loaded {len(self.entries):,} entries from "
            f"{len(jsonl_paths)} files; excluded={excluded:,}."
        )
        if not self.entries:
            raise ValueError("No training records remain after loading/filtering")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        gt_data = entry.get("ground_truth", {})
        entry_type = entry.get("data_type", entry.get("type", "affordance"))
        supervision = entry.get("supervision") or {}
        default_mask = entry_type in {"Affordance_Grounding", "affordance"}
        default_pose = entry_type in {"Pose_Regression", "pose"}
        allow_mask = bool(supervision.get("mask", default_mask))
        allow_pose = bool(supervision.get("pose", default_pose))
        allow_language = bool(supervision.get("language", True))

        # Load image
        img_path = self.data_root / entry["image"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read dataset image: {img_path}")

        H_orig, W_orig = bgr.shape[:2]

        # [CRITICAL-FIX] Augment returns flip state for GT synchronization
        is_flipped = False
        if self.augment:
            spatial_vqa = (
                entry_type in {"VQA", "vqa"}
                and gt_data.get("qa_type") == "spatial"
            )
            bgr, is_flipped = self._augment(
                bgr, allow_horizontal_flip=not spatial_vqa,
            )

        pv_dino = _preprocess_image(bgr, self.dino_size, DINO_MEAN, DINO_STD)
        pv_siglip = _preprocess_image(bgr, self.siglip_size, SIGLIP_MEAN, SIGLIP_STD)

        # Parse special tokens from conversations
        cleaned_convs, inline_mask_path, inline_pose_dict = extract_and_clean_special_tokens(
            entry["conversations"]
        )

        # 视角弱标签只监督 VA-Fusion 门控，不写入实验指令（论文式(16)）。
        object_class = gt_data.get("object_class", "")

        # 语言 CE 仅监督最后一个 assistant 目标序列 Y_i。
        input_ids, attention_mask, labels = tokenize_chat_with_assistant_labels(
            cleaned_convs, self.tokenizer, self.max_length,
        )

        if not allow_language:
            labels.fill_(-100)

        # Ground truth mask
        gt_mask = torch.zeros(1, self.mask_size, self.mask_size, dtype=torch.float32)
        has_mask = False
        mask_path_str = inline_mask_path or gt_data.get("mask_path")
        if allow_mask and mask_path_str:
            mask_full = self.data_root / mask_path_str
            mask_img = cv2.imread(str(mask_full), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # [CRITICAL-FIX] Apply same flip to mask
                if is_flipped:
                    mask_img = cv2.flip(mask_img, 1)
                # [CRITICAL-FIX] Use INTER_NEAREST to preserve binary mask edges
                mask_rs = cv2.resize(
                    mask_img, (self.mask_size, self.mask_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                gt_mask[0] = torch.from_numpy((mask_rs > 127).astype(np.float32))
                has_mask = True
        if allow_mask and not has_mask:
            raise FileNotFoundError(
                f"Mask supervision is enabled but the mask is missing/unreadable: "
                f"id={entry.get('id')} path={mask_path_str}"
            )

        # Ground truth pose
        gt_pose = torch.zeros(5, dtype=torch.float32)
        has_pose = False
        pose_src = inline_pose_dict or gt_data.get("pose")
        if allow_pose and pose_src:
            try:
                # [CRITICAL-FIX] Pass is_flipped to encode_pose
                gt_pose = encode_pose(pose_src, W_orig, H_orig, is_flipped=is_flipped)
                has_pose = True
            except Exception:
                if self.strict_data:
                    raise ValueError(
                        f"Invalid pose supervision: id={entry.get('id')} pose={pose_src}"
                    )
        if allow_pose and not has_pose:
            raise ValueError(
                f"Pose supervision is enabled but pose is missing: id={entry.get('id')}"
            )

        type_map = {
            "VQA": 0, "vqa": 0,
            "Affordance_Grounding": 1, "affordance": 1,
            "Pose_Regression": 2, "pose": 2,
        }
        type_id = type_map.get(entry_type, 1)

        # [VA-Fusion] 类别派生视角弱标签: 1=正视, 0=俯视, -1=未知 (训练时监督门控)
        # 推理时门控纯靠图像预测, 不依赖此标签
        vp_tag = get_viewpoint_tag(object_class)
        if vp_tag == "[Front-view]":
            view_label = 1
        elif vp_tag == "[Top-view]":
            view_label = 0
        else:
            view_label = -1

        return {
            "pixel_values_dino": pv_dino,
            "pixel_values_siglip": pv_siglip,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_mask": gt_mask,
            "gt_pose": gt_pose,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "has_pose": torch.tensor(has_pose, dtype=torch.bool),
            "type_id": torch.tensor(type_id, dtype=torch.long),
            "view_label": torch.tensor(view_label, dtype=torch.long),
        }

    def _augment(
        self, bgr: np.ndarray, allow_horizontal_flip: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        [CRITICAL-FIX] Returns (augmented_image, is_flipped) so caller can
        synchronize mask and pose ground truth.

        [OPT-20] Enhanced photometric augmentation for better generalization:
        - Horizontal flip (50%)
        - Brightness/contrast jitter
        - Gaussian blur
        - Color channel shuffle (rare, 10%)
        - Gaussian noise (for transparent objects)
        """
        is_flipped = False
        if allow_horizontal_flip and random.random() < 0.5:
            bgr = cv2.flip(bgr, 1)
            is_flipped = True

        # Photometric augmentation (does NOT change spatial layout)
        # 1. Brightness/contrast
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-25, 25)
        bgr = np.clip(alpha * bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 2. Gaussian blur
        if random.random() < 0.2:
            k = random.choice([3, 5])
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)

        # 3. [OPT-20] Slight color jitter (hue shift) for transparent object robustness
        if random.random() < 0.15:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hue_shift = random.randint(-10, 10)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 4. [OPT-20] Slight Gaussian noise for edge detection robustness
        if random.random() < 0.1:
            noise = np.random.normal(0, 5, bgr.shape).astype(np.float32)
            bgr = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return bgr, is_flipped


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder — [OPT-16] with scale-aware sampling option
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg, tokenizer, rank: int = 0, world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    data_root = cfg.data_root

    train_jsonl = getattr(cfg, "train_jsonl", "")
    val_jsonl = getattr(cfg, "val_jsonl", "")

    if train_jsonl and os.path.exists(train_jsonl):
        train_files = [train_jsonl]
        val_files = [val_jsonl] if val_jsonl and os.path.exists(val_jsonl) else []
        use_explicit_split = True
        if getattr(cfg, "strict_data", True) and not val_files:
            raise FileNotFoundError(
                f"Explicit validation JSONL is required but missing: {val_jsonl}"
            )
    elif cfg.use_combined_jsonl:
        train_files = [os.path.join(data_root, "CoT", "combined_dataset.jsonl")]
        val_files = []
        use_explicit_split = False
    else:
        train_files = [
            os.path.join(data_root, "CoT", "affordance_dataset.jsonl"),
            os.path.join(data_root, "CoT", "pose_dataset.jsonl"),
            os.path.join(data_root, "CoT", "vqa_dataset.jsonl"),
        ]
        val_files = []
        use_explicit_split = False

    train_dataset = VLADataset(
        jsonl_paths=train_files,
        data_root=data_root,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_len,
        dino_size=cfg.dino_size,
        siglip_size=cfg.siglip_size,
        mask_size=cfg.img_size,
        augment=True,
        exclude_classes=getattr(cfg, "exclude_classes", []),
        strict_data=getattr(cfg, "strict_data", True),
    )

    if use_explicit_split and val_files:
        val_dataset = VLADataset(
            jsonl_paths=val_files,
            data_root=data_root,
            tokenizer=tokenizer,
            max_length=cfg.max_seq_len,
            dino_size=cfg.dino_size,
            siglip_size=cfg.siglip_size,
            mask_size=cfg.img_size,
            augment=False,
            exclude_classes=getattr(cfg, "exclude_classes", []),
            strict_data=getattr(cfg, "strict_data", True),
        )
    else:
        n_val = max(1, int(0.10 * len(train_dataset)))
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # ── Samplers ──────────────────────────────────────────────────────────
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
