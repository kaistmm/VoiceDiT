# Codes adapted from https://github.com/heatz123/naturalspeech/blob/main/models/models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from voicedit.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from voicedit import monotonic_align

from voicedit.text_encoder import TextEncoder
from text.symbols import symbols

class DiTWrapper(nn.Module):
    def __init__(self, dit, cond_speaker=False, concat_y=False, uncond_prob=0.0, dit_arch=True):
        super().__init__()
        self.dit = dit
        self.cond_speaker = cond_speaker
        self.uncond_prob = uncond_prob
        self.n_feats = 64
        self.concat_y = concat_y
        self.dit_arch = dit_arch

        self.text_encoder = TextEncoder(len(symbols)+1, n_feats = 64, n_channels = 192, 
                                   filter_channels = 768, filter_channels_dp = 256, n_heads = 2, 
                                   n_layers = 4, kernel_size = 3, p_dropout = 0.1, window_size = 4, spk_emb_dim = self.n_feats, cond_spk=cond_speaker)

        if cond_speaker:
            self.spk_embedding = torch.nn.Sequential(torch.nn.Linear(192, 192 * 4), nn.ReLU(),
                                                     torch.nn.Linear(192 * 4, self.n_feats))
        if concat_y:
            self.proj = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), value=0),
                nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(),
                nn.ConstantPad2d((0, 1, 0, 1), value=0),
                nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2)),
            )
        
        if not dit_arch:
            self.class_embedding = nn.Linear(512, 1024)
        
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(8, 256, 16)) if concat_y 
                             else nn.Parameter(torch.randn(1, self.n_feats) / self.n_feats ** 0.5))

        
    def forward(self, sample, timestep, text, text_lengths=None, cond=None, y=None, y_lengths=None, y_latent=None, speech_starts=None, spk=None, train_text_encoder=True, **kwargs):
        
        if train_text_encoder:
            if spk is not None:
                spk = self.spk_embedding(spk)

            # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
            mu_x, logw, x_mask = self.text_encoder(text, text_lengths, spk)
            y_max_length = y.shape[-1]

            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
            
            # Compute loss between predicted log-scaled durations and those obtained from MAS
            logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
            dur_loss = duration_loss(logw, logw_, text_lengths)

            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            
            # Compute loss between aligned encoder outputs and mel-spectrogram
            prior_loss = torch.sum(0.5 * ((y - mu_y.transpose(1,2)) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
            # spk_emb = spk_emb.unsqueeze(1).repeat(1, mu_y.shape[1], 1)
            # mu_y_vae = torch.stack([mu_y, spk_emb], dim=1)

            if self.concat_y:
                condition_y = self.proj(mu_y.unsqueeze(1))
                
                # y_latent_lengths = torch.ceil(y_lengths / 4).long()
                # y_latent_mask = sequence_mask(y_latent_lengths, y_latent.shape[-2]).unsqueeze(1).to(mu_y_latent.dtype)
                # latent_prior_loss = torch.sum(0.5 * ((y_latent - mu_y_latent) ** 2 + math.log(2 * math.pi)) * y_latent_mask.unsqueeze(-1))
                # prior_loss += latent_prior_loss / (torch.sum(y_latent_mask) * y_latent.shape[1] * y_latent.shape[-1])
                
                condition_y = torch.nn.functional.pad(condition_y, (0, 0, 0, sample.shape[-2] - condition_y.shape[-2]), value=0)
            else:
                condition_y = mu_y.unsqueeze(1)
        
        else:
            
            with torch.no_grad():
                mu_y, y_mask = self.process_content(text, text_lengths, spk)
                
                if self.concat_y:
                    condition_y = self.proj(mu_y.unsqueeze(1))
                    
                    # y_latent_lengths = torch.ceil(y_lengths / 4).long()
                    # y_latent_mask = sequence_mask(y_latent_lengths, y_latent.shape[-2]).unsqueeze(1).to(mu_y_latent.dtype)
                    # latent_prior_loss = torch.sum(0.5 * ((y_latent - mu_y_latent) ** 2 + math.log(2 * math.pi)) * y_latent_mask.unsqueeze(-1))
                    # prior_loss += latent_prior_loss / (torch.sum(y_latent_mask) * y_latent.shape[1] * y_latent.shape[-1])
                    
                    # condition_temp = torch.zeros(sample.size()).to(condition_y.device, dtype=condition_y.dtype)
                    
                    if speech_starts is not None and torch.any(speech_starts):
                        condition_temp = []
                        for start in speech_starts:
                            condition_temp.append(torch.nn.functional.pad(condition_y[0], (0, 0, start, sample.shape[-2] - condition_y[0].shape[-2] - start), value=0))
                        condition_y = torch.stack(condition_temp, dim=0)
                        # condition_temp_y = torch.zeros_like(sample)
                        # indices = speech_starts.unsqueeze(1).unsqueeze(2).unsqueeze(3) + torch.arange(condition_y.shape[-2]).unsqueeze(0).unsqueeze(1).unsqueeze(3).to(speech_starts.device)
                        # condition_temp_y.scatter_(2, indices.expand(condition_y.size()), condition_y)
                        # condition_y = condition_temp_y
                    else:
                        condition_y = torch.nn.functional.pad(condition_y, (0, 0, 0, sample.shape[-2] - condition_y.shape[-2]), value=0)
                else:
                    condition_y = mu_y.unsqueeze(1)

        if self.concat_y and self.uncond_prob > 0.0:
            drop_ids = torch.rand(condition_y.shape[0]).cuda() < self.uncond_prob
            condition_y = torch.where(drop_ids[:, None, None, None], self.y_embedding, condition_y)
        
        if self.dit_arch:
            sample = self.dit(
                sample, 
                timestep,
                y=condition_y,
                mask=y_mask.long() if not self.concat_y else None,
                class_labels=cond,
            )
        else:
            sample = torch.cat([sample, condition_y], dim=1)
            cond = self.class_embedding(cond)
            sample = self.dit(
                sample,
                timestep,
                class_labels=cond,
            ).sample
        
        if sample.isnan().any():
            print("NAN detected from sample")

        if train_text_encoder:
            return dict(
                x=sample,
                dur_loss=dur_loss,
                prior_loss=prior_loss,
            )
        else:
            return sample
    
    def process_content(self, x, x_lengths, spk=None, length_scale=1.0):
        
        if spk is not None:
            spk = self.spk_embedding(spk)
        
        if spk.shape[0] != x.shape[0]:
            spk = spk.repeat(x.shape[0], 1)
    
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.text_encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        
        # spk_emb = spk_emb.unsqueeze(1).repeat(1, mu_y.shape[1], 1)
        # mu_y_vae = torch.stack([mu_y, spk_emb], dim=1)
        
        # if self.concat_y == 'vae':
        #     assert vae is not None
        #     mu_y_latent = vae.encode(mu_y.unsqueeze(1)).latent_dist.sample()
        #     mu_y_latent = mu_y_latent * vae.config.scaling_factor
        # if self.concat_y:
        #     condition_y = self.proj(mu_y.unsqueeze(1))
        # else:
        #     condition_y = mu_y.unsqueeze(1)
    
        return mu_y, y_mask
    