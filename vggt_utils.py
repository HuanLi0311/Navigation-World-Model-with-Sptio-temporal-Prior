# utils.py
# ---------------------------------------------------------
# Safe VGGT integration for NWM (DDP / multi-node ready)
# ---------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# Add InfiniteVGGT to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "InfiniteVGGT", "src")
)

from vggt.models.vggt import VGGT


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# ---------------------------------------------------------
# Load VGGT (CPU ONLY)
# ---------------------------------------------------------

def load_vggt_model(
    checkpoint_path,
    img_size=518,
    patch_size=14,
    embed_dim=1024,
):
    """
    IMPORTANT:
    - Always load VGGT on CPU
    - Only move to GPU when extracting features
    """

    model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        if is_rank0():
            print(f"[VGGT] loading checkpoint from {checkpoint_path}")

        # ⚠️ map_location 必须是函数 or device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    else:
        if is_rank0():
            print(f"[VGGT] WARNING: checkpoint not found, using random init")

    model.eval()
    model.requires_grad_(False)
    return model


# ---------------------------------------------------------
# Feature extraction (TEMPORARY GPU)
# ---------------------------------------------------------

@torch.no_grad()
def extract_vggt_features(
    vggt_model,
    images,                     # [B, S, C, H, W] in [-1, 1]
    vggt_img_size=518,
):
    """
    VGGT is moved to GPU only inside this function.
    """

    device = images.device

    # 1. move model to GPU temporarily
    vggt_model = vggt_model.to(device)

    # 2. normalize
    images = (images + 1.0) / 2.0
    B, S, C, H, W = images.shape

    # 3. resize if needed
    if H != vggt_img_size or W != vggt_img_size:
        images = images.flatten(0, 1)
        images = F.interpolate(
            images,
            size=(vggt_img_size, vggt_img_size),
            mode="bilinear",
            align_corners=False,
        )
        images = images.unflatten(0, (B, S))

    # 4. forward
    aggregated_tokens_list, _ = vggt_model.aggregator(images)

    tokens = aggregated_tokens_list[-1]  # [B, S, N, 2*D]

    # 5. merge frame/global
    B, S, N, D2 = tokens.shape
    tokens = tokens.view(B, S, N, 2, D2 // 2).mean(dim=3)

    # 6. IMPORTANT: detach graph
    tokens = tokens.detach()

    # 7. move VGGT back to CPU
    vggt_model.to("cpu")
    torch.cuda.empty_cache()

    return tokens


# ---------------------------------------------------------
# Pruning
# ---------------------------------------------------------

def prune_kv_cache(tokens, keep_ratio=0.5, method="attention"):
    """
    tokens: [B, S, N, D]
    """
    B, S, N, D = tokens.shape
    num_keep = max(1, int(N * keep_ratio))

    if method == "first":
        return tokens[:, :, :num_keep, :]

    elif method == "random":
        idx = torch.randperm(N, device=tokens.device)[:num_keep]
        idx, _ = torch.sort(idx)
        return tokens[:, :, idx, :]

    elif method == "attention":
        importance = tokens.norm(dim=-1).mean(dim=1)  # [B, N]
        _, idx = torch.topk(importance, k=num_keep, dim=-1)
        idx, _ = torch.sort(idx)

        idx = idx.unsqueeze(1).unsqueeze(-1).expand(B, S, num_keep, D)
        return torch.gather(tokens, dim=2, index=idx)

    else:
        raise ValueError(f"Unknown prune method: {method}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

@torch.no_grad()
def prepare_kv_cache(
    vggt_model,
    images,
    keep_ratio=0.5,
    prune_method="attention",
    vggt_img_size=518,
):
    """
    Correct usage pattern:
    - VGGT only touches GPU inside this function
    - KV cache stays on GPU
    """

    # 1. extract
    tokens = extract_vggt_features(
        vggt_model,
        images,
        vggt_img_size=vggt_img_size,
    )

    # 2. prune
    tokens = prune_kv_cache(tokens, keep_ratio, prune_method)

    return tokens
