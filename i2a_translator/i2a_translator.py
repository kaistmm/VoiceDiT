import math
import random
from tqdm.auto import tqdm
import json
import os

import torch
import torch.nn.functional as F
from torch import nn, einsum

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding
# pip install 'pydantic>=2'
# pip install 'einops>=0.7.0'
# pip install 'rotary-embedding-torch'
# pip install 'pytorch-warmup'

# helper functions

def exists(val):
    return val is not None

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])

# tensor helpers

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion helper functions

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    # attempting to correct nan gradients when learned variance is turned on
    # in the setting of deepspeed fp16
    eps = 1e-12 if x.dtype == torch.float32 else 1e-3

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps = eps)
    log_one_minus_cdf_min = log(1. - cdf_min, eps = eps)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta, eps = eps)))

    return log_probs

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype = torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma = 0., p2_loss_weight_k = 1):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device = self.betas.device, dtype = torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def calculate_v(self, x_start, t, noise = None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)

# diffusion prior

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# mlp

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor = 2.,
        depth = 2,
        norm = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

# relative positional bias for transformer

# class RelPosBias(nn.Module):
#     def __init__(
#         self,
#         causal = False,
#         heads = 8,
#         num_buckets = 32,
#         max_distance = 128,
#     ):
#         super().__init__()
#         self.causal = causal
#         self.num_buckets = num_buckets
#         self.max_distance = max_distance
#         self.relative_attention_bias = nn.Embedding(num_buckets, heads)

#     @staticmethod
#     def _relative_position_bucket(
#         relative_position,
#         causal = True,
#         num_buckets = 32,
#         max_distance = 128
#     ):
#         ret = 0
#         n = -relative_position
#         if not causal:
#             num_buckets //= 2
#             ret += (n < 0).long() * num_buckets
#             n = torch.abs(n)
#         else:
#             n = torch.max(n, torch.zeros_like(n))

#         max_exact = num_buckets // 2
#         is_small = n < max_exact

#         val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
#         val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

#         ret += torch.where(is_small, n, val_if_large)
#         return ret

#     def forward(self, i, j, *, device):
#         q_pos = torch.arange(i, dtype = torch.long, device = device)
#         k_pos = torch.arange(j, dtype = torch.long, device = device)
#         rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
#         rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
#         values = self.relative_attention_bias(rp_bucket)
#         return rearrange(values, 'i j h -> h i j')

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

# feedforward

class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# Attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        rotary_emb = None,
        cosine_sim = True,
        cosine_sim_scale = 16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# class Transformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         depth,
#         dim_head = 64,
#         heads = 8,
#         ff_mult = 4,
#         norm_in = False,
#         norm_out = True,
#         attn_dropout = 0.,
#         ff_dropout = 0.,
#         final_proj = True,
#         normformer = False,
#         rotary_emb = True,
#         causal = True,
#     ):
#         super().__init__()
#         self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

#         self.causal = causal

#         self.rel_pos_bias = RelPosBias(causal = causal, heads = heads)

#         rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
#                 FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
#             ]))

#         self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
#         self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

#     def forward(self, x):
#         n, device = x.shape[1], x.device

#         x = self.init_norm(x)

#         attn_bias = self.rel_pos_bias(n, n + 1, device = device)

#         for attn, ff in self.layers:
#             x = attn(x, attn_bias = attn_bias) + x
#             x = ff(x) + x

#         out = self.norm(x)
#         return self.project_out(out)

# class DiffusionPriorNetwork(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_timesteps = None,
#         num_time_embeds = 1,
#         num_image_embeds = 1,
#         num_text_embeds = 1,
#         max_text_len = 256,
#         self_cond = False,
#         **kwargs
#     ):
#         super().__init__()
#         self.dim = dim

#         self.num_time_embeds = num_time_embeds
#         self.num_image_embeds = num_image_embeds
#         self.num_text_embeds = num_text_embeds

#         self.to_text_embeds = nn.Sequential(
#             nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
#             Rearrange('b (n d) -> b n d', n = num_text_embeds)
#         )

#         self.continuous_embedded_time = not exists(num_timesteps)

#         self.to_time_embeds = nn.Sequential(
#             nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
#             Rearrange('b (n d) -> b n d', n = num_time_embeds)
#         )

#         self.to_image_embeds = nn.Sequential(
#             nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
#             Rearrange('b (n d) -> b n d', n = num_image_embeds)
#         )

#         self.learned_query = nn.Parameter(torch.randn(dim))
#         self.transformer = Transformer(dim = dim, **kwargs)

#         # dalle1 learned padding strategy

#         self.max_text_len = max_text_len

#         self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
#         self.null_image_embed = nn.Parameter(torch.randn(1, dim))

#         # whether to use self conditioning, Hinton's group's new ddpm technique

#         self.self_cond = self_cond

#     def forward_with_cond_scale(
#         self,
#         *args,
#         cond_scale = 1.,
#         **kwargs
#     ):
#         logits = self.forward(*args, **kwargs)

#         if cond_scale == 1:
#             return logits

#         # null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
#         null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 0., **kwargs)
#         return null_logits + (logits - null_logits) * cond_scale

#     def forward(
#         self,
#         image_embed,
#         diffusion_timesteps,
#         *,
#         text_embed,
#         text_encodings = None,
#         self_cond = None,
#         text_cond_drop_prob = 0.,
#         image_cond_drop_prob = 0.
#     ):
#         batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

#         num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds

#         # setup self conditioning

#         if self.self_cond:
#             self_cond = default(self_cond, lambda: torch.zeros(batch, self.dim, device = device, dtype = dtype))
#             self_cond = rearrange(self_cond, 'b d -> b 1 d')

#         # in section 3.3, last paragraph
#         # "... we craft a learnable token of 512-d whose output from the Transformer is considered as the recovered audio embedding.
#         #  We then take the time embedding, noisy audio embedding, as well as the visual condition as three other tokens of the same shape"

#         text_embed = self.to_text_embeds(text_embed)
#         image_embed = self.to_image_embeds(image_embed)

#         # classifier free guidance masks

#         text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob, device = device)
#         text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

#         image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
#         image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

#         # mask out text embeddings with null text embeddings

#         null_text_embeds = self.null_text_embeds.to(text_embed.dtype)

#         text_embed = torch.where(
#             text_keep_mask,
#             text_embed,
#             null_text_embeds
#         )

#         # mask out image embeddings with null image embeddings

#         null_image_embed = self.null_image_embed.to(image_embed.dtype)

#         image_embed = torch.where(
#             image_keep_mask,
#             image_embed,
#             null_image_embed
#         )

#         # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
#         # but let's just do it right

#         if self.continuous_embedded_time:
#             diffusion_timesteps = diffusion_timesteps.type(dtype)

#         time_embed = self.to_time_embeds(diffusion_timesteps)

#         learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

#         if self.self_cond:
#             learned_queries = torch.cat((self_cond, learned_queries), dim = -2)
        

#         tokens = torch.cat((
#             learned_queries,
#             time_embed,
#             image_embed,
#             text_embed
#         ), dim = -2)

#         # attend

#         tokens = self.transformer(tokens)

#         # get learned query, which should predict the audio embedding (per DDPM timestep)

#         pred_image_embed = tokens[..., 0, :]

#         return pred_image_embed

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


class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip = None,
        image_embed_dim = None,
        image_size = None,
        image_channels = 3,
        timesteps = 1000,
        sample_timesteps = None,
        cond_drop_prob = 0.,
        text_cond_drop_prob = None,
        image_cond_drop_prob = None,
        loss_type = "l2",
        predict_x_start = True,
        predict_v = False,
        beta_schedule = "cosine",
        condition_on_text_encodings = True,  # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm = False,       # whether to l2norm clamp the image embed at each denoising iteration (analogous to -1 to 1 clipping for usual DDPMs)
        sampling_final_clamp_l2norm = False, # whether to l2norm the final image embedding output (this is also done for images in ddpm)
        training_clamp_l2norm = False,
        init_image_embed_l2norm = False,
        image_embed_scale = None,            # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        clip_adapter_overrides = dict()
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        self.noise_scheduler = NoiseScheduler(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        assert exists(image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
        self.clip = None

        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)

        assert net.dim == self.image_embed_dim, f'your diffusion prior network has a dimension of {net.dim}, but you set your image embedding dimension (keyword image_embed_dim) on DiffusionPrior to {self.image_embed_dim}'
        assert not exists(clip) or clip.dim_latent == self.image_embed_dim, f'you passed in a CLIP to the diffusion prior with latent dimensions of {clip.dim_latent}, but your image embedding dimension (keyword image_embed_dim) for the DiffusionPrior was set to {self.image_embed_dim}'

        self.channels = default(image_channels, lambda: clip.image_channels)

        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        self.can_classifier_guidance = self.text_cond_drop_prob > 0. or self.image_cond_drop_prob > 0.
        self.condition_on_text_encodings = condition_on_text_encodings

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v # takes precedence over predict_x_start

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132

        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker

        self.register_buffer('_dummy', torch.tensor([True]), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(self, x, t, text_cond, self_cond = None, clip_denoised = False, cond_scale = 1.):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = self.net.forward_with_cond_scale(x, t, cond_scale = cond_scale, self_cond = self_cond, **text_cond)

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1.):
        batch, device = shape[0], self.device

        image_embed = torch.randn(shape, device = device)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape, text_cond, *, timesteps, eta = 1., cond_scale = 1., progress = True):
        batch, device, alphas, total_timesteps = shape[0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        times = torch.linspace(-1., total_timesteps, steps = timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device = device)

        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=not progress):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None

            pred = self.net.forward_with_cond_scale(image_embed, time_cond, self_cond = self_cond, cond_scale = cond_scale, **text_cond)

            # derive x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t = time_cond, v = pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t = time_cond, noise = pred)

            # clip x0 before maybe predicting noise

            if not self.predict_x_start:
                x_start.clamp_(-1., 1.)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t = time_cond, x0 = x_start)

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.

            image_embed = x_start * alpha_next.sqrt() + \
                          c1 * noise + \
                          c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        image_embed = normalized_image_embed / self.image_embed_scale
        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale = 1.):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device = device)

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc = 'sampling loop time step', total = self.noise_scheduler.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device = device, dtype = torch.long), text_cond = text_cond, cond_scale = cond_scale)
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        text,
        num_samples_per_batch = 2,
        cond_scale = 1.,
        timesteps = None
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond, cond_scale = cond_scale, timesteps = timesteps)

        # retrieve original unscaled image embed

        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r = num_samples_per_batch)

        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k = 1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d = image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(
        self,
        text = None,
        image = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed), 'either text or text embedding must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_path):
        config_path = os.path.join(pretrained_path, 'prior_config.json')
        ckpt_path = os.path.join(pretrained_path, 'best_checkpoint.pth')
        with open(config_path) as f:
            config = json.load(f)
        prior_config = config['prior']
        net_config = prior_config.pop('net')
        diffusion_prior_network = DiffusionPriorNetwork(**net_config)
        model = cls(net = diffusion_prior_network, **prior_config)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'], strict=True)
        return model


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)