import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from dalle2_pytorch.dalle2_pytorch import (
    LayerNorm,
    FeedForward,
    Attention,
    RotaryEmbedding,
    prob_mask_like,
    exists,
    repeat,
    SinusoidalPosEmb,
    MLP,
    default
)

class RelPosBias(nn.Module):
    def __init__(
        self,
        causal = False,
        heads = 8,
        num_buckets = 32,
        max_distance = 128,
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal = True,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.causal = causal

        self.rel_pos_bias = RelPosBias(causal = causal, heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class DiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        text_enc_dim = None,
        num_timesteps = None,
        num_time_embeds = 1,
        num_image_embeds = 1,
        num_text_embeds = 1,
        max_text_len = 256,
        self_cond = False,
        causal = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.text_enc_dim = text_enc_dim

        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        self.to_text_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_text_embeds)
        )

        if exists(text_enc_dim) and text_enc_dim != dim:
            self.to_text_encodings = nn.Sequential(
                nn.Linear(text_enc_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        # else:
        #     self.to_text_encodings = nn.Identity()

        self.continuous_embedded_time = not exists(num_timesteps)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_image_embeds)
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim = dim, causal = causal, **kwargs)

        # dalle1 learned padding strategy

        self.max_text_len = max_text_len

        if exists(text_enc_dim):
            self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))

        # whether to use self conditioning, Hinton's group's new ddpm technique

        self.self_cond = self_cond

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        # null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 0., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings = None,
        self_cond = None,
        text_cond_drop_prob = 0.,
        image_cond_drop_prob = 0.
    ):
        batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        
        num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds
        
        # setup self conditioning

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros(batch, self.dim, device = device, dtype = dtype))
            self_cond = rearrange(self_cond, 'b d -> b 1 d')

        # in section 3.3, last paragraph
        # "... we craft a learnable token of 512-d whose output from the Transformer is considered as the recovered audio embedding.
        #  We then take the time embedding, noisy audio embedding, as well as the visual condition as three other tokens of the same shape"

        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)

        # classifier free guidance masks

        text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob, device = device)
        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')
        
        if not exists(text_encodings):
            
            text_encodings = torch.empty((batch, 0, dim), device = device, dtype = dtype)
        
        else:

            mask = torch.any(text_encodings != 0., dim = -1)

            text_encodings = self.to_text_encodings(text_encodings)
            text_encodings = text_encodings * mask.unsqueeze(-1)

            # replace any padding in the text encodings with learned padding tokens unique across position

            text_encodings = text_encodings[:, :self.max_text_len]
            mask = mask[:, :self.max_text_len]

            text_len = text_encodings.shape[-2]
            remainder = self.max_text_len - text_len
        
            if remainder > 0:
                text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value = 0.)
                mask = F.pad(mask, (0, remainder), value = False)

            # mask out text encodings with null encodings

            null_text_encodings = self.null_text_encodings.to(text_encodings.dtype)
            
            text_encodings = torch.where(
                rearrange(mask, 'b n -> b n 1').clone() & text_keep_mask,
                text_encodings,
                null_text_encodings
            )

        # mask out text embeddings with null text embeddings

        null_text_embeds = self.null_text_embeds.to(text_embed.dtype)

        text_embed = torch.where(
            text_keep_mask,
            text_embed,
            null_text_embeds
        )

        # mask out image embeddings with null image embeddings

        null_image_embed = self.null_image_embed.to(image_embed.dtype)

        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed
        )

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        if self.self_cond:
            learned_queries = torch.cat((self_cond, learned_queries), dim = -2)
        
        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            image_embed,
            learned_queries,
        ), dim = -2)

        # attend
        
        tokens = self.transformer(tokens)

        # get learned query, which should predict the audio embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed
    