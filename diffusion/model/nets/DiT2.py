# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

import sys
sys.path.insert(1, os.getcwd())

from diffusion.model.utils import auto_grad_checkpoint, to_2tuple, PatchEmbed
from diffusion.model.nets.DiT_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer, modulate
# from diffusion.utils.logger import get_root_logger


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, cross_class=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        if cross_class:
            self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.window_size = window_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t, c=None, mask=None, **kwargs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if hasattr(self, 'cross_attn'):
            x = x + self.cross_attn(x, c, mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


#############################################################################
#                                 Core DiT Model                                #
#################################################################################
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, concat_y=False, cross_class=False, class_dropout_prob=0.0, pred_sigma=True, projection_class_embeddings_input_dim=512, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        self.cross_class = cross_class

        self.concat_y = concat_y
        if isinstance(input_size, int):
            input_size = to_2tuple(input_size)
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels * 2 if concat_y else in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = nn.Linear(projection_class_embeddings_input_dim, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size[0] // patch_size, input_size[1] // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                          cross_class=cross_class)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def forward(self, x, timestep, class_labels, y, mask=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, C, H, W) tensor of caption labels
        class_labels: (N, C') tensor of class labels
        """
        
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        if self.concat_y:
            x = torch.cat([x, y], dim=1)
        x = self.x_embedder(x) + pos_embed          # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)               # (N, D)
        class_embed = self.c_embedder(class_labels)
        t0 = t + class_embed
        if self.cross_class:
            class_lens = [1] * class_embed.shape[0]
            class_embed = class_embed.unsqueeze(1).view(1, -1, x.shape[-1])
        else:
            class_lens = None
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, t0, class_embed, class_lens)
        x = self.final_layer(x, t0)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, class_labels=None, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask, class_labels)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, class_labels=None, guidance="single", **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if guidance == "dual":
            num_chunk = 4
        elif guidance == "single":
            num_chunk = 2
        chunk = x[: len(x) // num_chunk]
        combined = torch.cat([chunk] * num_chunk, dim=0)
        model_out = self.forward(combined, timestep, class_labels, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps = eps.chunk(num_chunk, dim=0)
        if guidance == "dual":
            pred_eps = eps[0] + cfg_scale[0] * (eps[1] - eps[3]) + cfg_scale[1] * (eps[2] - eps[3])
        elif guidance == "single":
            pred_eps = eps[1] + cfg_scale * (eps[0] - eps[1])
        eps = torch.cat([pred_eps] * num_chunk, dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p1, p2 = self.x_embedder.patch_size
        h, w = self.x_embedder.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p1, w * p2))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if hasattr(block, 'cross_attn'):
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


DiT_models = {
    'DiT_XL/2': DiT_XL_2,
    'DiT_L/2': DiT_L_2,
    'DiT_B/2': DiT_B_2,
    'DiT_S/2': DiT_S_2,
}

if __name__ == '__main__':

    from thop import profile, clever_format

    device = "cuda:0"
    model = DiT_models['DiT_L/2'](
        input_size = (256, 16),
        in_channels = 1,
        projection_class_embeddings_input_dim=512,
        concat_y = True,
        cross_class = True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    x = torch.randn(1, 1, 256, 16).to(device)
    timestep = torch.Tensor([999]).to(device)
    y = torch.randn(1, 1, 256, 16).to(device)
    mask = torch.ones(1, 1, 1, 1).to(device)
    class_labels = torch.randn(1, 512).to(device)

    macs, params = profile(model, inputs=(x, timestep, class_labels, y))
    flops = macs * 2
    macs, params, flops = clever_format([macs, params, flops], "%.4f")
    print("Params:", params)
    print("MACs:", macs)
    print("FLOPs:", flops)
    import pdb; pdb.set_trace()