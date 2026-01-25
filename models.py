# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import sys
import os

# Add InfiniteVGGT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'InfiniteVGGT', 'src'))
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size//3
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size -2*hsize, frequency_embedding_size)

    def forward(self, xya):
        return torch.cat([self.x_emb(xya[...,0:1]), self.y_emb(xya[...,1:2]), self.angle_emb(xya[...,2:3])], dim=-1)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

class AttentionWithRoPE(nn.Module):
    """
    Multi-head attention with 2D RoPE support (InfiniteVGGT style).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, rope_module=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope_module = rope_module
    
    def forward(self, x, positions=None):
        """
        Args:
            x: (B, N, C) input tokens
            positions: (B, N, 2) spatial positions for each token (y, x coordinates)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Apply RoPE if available
        if self.rope_module is not None and positions is not None:
            # InfiniteVGGT expects: (batch, n_heads, n_tokens, dim)
            q = self.rope_module(q, positions)
            k = self.rope_module(k, positions)
        
        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Modified to use unified self-attention on concatenated tokens instead of separate self and cross attention.
    Optionally includes cross-attention with KV cache from VGGT.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_size=2, rope_module=None, use_kv_cross_attn=False, **block_kwargs):
        super().__init__()
        self.context_size = context_size
        self.use_kv_cross_attn = use_kv_cross_attn
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithRoPE(hidden_size, num_heads=num_heads, qkv_bias=True, rope_module=rope_module)
        
        # KV cache cross-attention (only for layers 16+)
        if use_kv_cross_attn:
            self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.kv_cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # AdaLN modulation: 6 params for self-attn and MLP, +3 for KV cross-attn if enabled
        num_modulation_params = 9 if use_kv_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_modulation_params * hidden_size, bias=True)
        )

    def forward(self, x, c, positions=None, kv_cache=None):
        """
        Args:
            x: (B, N, D) - main tokens (context + current frame)
            c: (B, D) - conditioning
            positions: (B, N, 2) - spatial positions for RoPE
            kv_cache: (B, M, D) - KV cache from VGGT (optional)
        """
        if self.use_kv_cross_attn:
            shift_msa, scale_msa, gate_msa, shift_kv, scale_kv, gate_kv, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention on all tokens (context + current frame) with RoPE
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), positions=positions)
        
        # Cross-attention with KV cache (if enabled and cache is provided)
        if self.use_kv_cross_attn and kv_cache is not None:
            x_normed = modulate(self.norm_kv(x), shift_kv, scale_kv)
            x = x + gate_kv.unsqueeze(1) * self.kv_cross_attn(
                query=x_normed,
                key=kv_cache,
                value=kv_cache,
                need_weights=False
            )[0]
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        kv_cache_dim=None,  # Dimension of VGGT KV cache
        kv_cross_attn_start_layer=6,  # Start KV cross-attention from this layer
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.kv_cross_attn_start_layer = kv_cross_attn_start_layer
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        
        # Initialize 2D RoPE (InfiniteVGGT style)
        self.rope_2d = RotaryPositionEmbedding2D(frequency=100.0, scaling_factor=1.0)
        self.position_getter = PositionGetter()
        
        # KV cache projection layer (if VGGT features are used)
        if kv_cache_dim is not None and kv_cache_dim != hidden_size:
            self.kv_cache_proj = nn.Linear(kv_cache_dim, hidden_size)
        else:
            self.kv_cache_proj = None
        
        # Build blocks: layers >= kv_cross_attn_start_layer have KV cross-attention
        self.blocks = nn.ModuleList([
            CDiTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio, 
                context_size=self.context_size, 
                rope_module=self.rope_2d,
                use_kv_cross_attn=(i >= kv_cross_attn_start_layer)
            ) 
            for i in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize KV cache projection if exists
        if self.kv_cache_proj is not None:
            nn.init.xavier_uniform_(self.kv_cache_proj.weight)
            nn.init.constant_(self.kv_cache_proj.bias, 0)


        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
            
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t, kv_cache=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        x_cond: (N, context_size, C, H, W) tensor of context frames
        rel_t: (N,) tensor of relative timesteps
        kv_cache: (N, M, D_kv) tensor of pruned KV cache from VGGT (optional)
        """
        batch_size = x.shape[0]
        num_patches = self.x_embedder.num_patches
        h = w = int(num_patches ** 0.5)
        
        # Embed current frame (no positional encoding added here, RoPE is applied in attention)
        x = self.x_embedder(x)  # (N, num_patches, D)
        
        # Embed context frames
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1]))  # (N, context_size, num_patches, D)
        x_cond = x_cond.flatten(1, 2)  # (N, context_size * num_patches, D)
        
        # Concatenate context and current frame tokens
        x = torch.cat([x_cond, x], dim=1)  # (N, (context_size + 1) * num_patches, D)
        
        # Generate positions for all tokens (context + current)
        # Each frame has the same spatial layout, so we repeat positions
        positions = self.position_getter(batch_size, h, w, x.device)  # (N, num_patches, 2)
        # Repeat for context_size + 1 frames
        positions = positions.unsqueeze(1).repeat(1, self.context_size + 1, 1, 1)  # (N, context_size+1, num_patches, 2)
        positions = positions.flatten(1, 2)  # (N, (context_size+1)*num_patches, 2)
        
        # Project KV cache to match hidden dimension if needed
        if kv_cache is not None and self.kv_cache_proj is not None:
            kv_cache = self.kv_cache_proj(kv_cache)  # (N, M, D)
        
        # Prepare conditioning
        t = self.t_embedder(t[..., None])
        y = self.y_embedder(y) 
        time_emb = self.time_embedder(rel_t[..., None])
        c = t + time_emb + y  # (N, D)

        # Pass through transformer blocks with unified self-attention and RoPE
        # Layers >= kv_cross_attn_start_layer also use KV cache cross-attention
        for block in self.blocks:
            x = block(x, c, positions=positions, kv_cache=kv_cache)
        
        # Extract only the current frame tokens for final prediction
        x = x[:, -num_patches:]  # (N, num_patches, D)
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}
