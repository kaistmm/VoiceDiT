import os
import yaml
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from copy import deepcopy
import random

from transformers import (
    ClapModel,
    ClapProcessor,
    CLIPModel,
    CLIPProcessor,
    SpeechT5HifiGan,
)
from diffusers import (
    AutoencoderKL, 
    DDIMScheduler,
    UNet2DModel,
)
try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download

from .utils import sequence_mask, generate_path, fix_len_compatibility
from espnet2.bin.spk_inference import Speech2Embedding
from i2a_translator import DiffusionPrior

from diffusion.model.nets.DiT2 import DiT_models
from diffusion.model.nets.DiT import DiT_cross_models
from diffusion import create_diffusion

from text import text_to_sequence, cmudict
from text.symbols import symbols
from .modules import DiTWrapper
from .utils import intersperse

# helper function
def exists(val):
    return val is not None
    

class VoiceDiTPipeline():
    def __init__(
        self,
        ckpt_path,
        v2a_ckpt_path = None,
        t2a_ckpt_path = None,
        device = None,
        cmudict_path='voicedit/cmu_dictionary',
        male_voice = None,
        female_voice = None,
    ):
        
        self.vae = AutoencoderKL.from_pretrained("cvssp/audioldm-m-full", subfolder="vae").eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm-m-full", subfolder="vocoder").eval()
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.speech2spk_embed = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_ecapa_wavlm_joint", device=str(device))
        self.cmudict = cmudict.CMUDict(cmudict_path)
        
        if exists(v2a_ckpt_path):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.v2a_mapper = DiffusionPrior.from_pretrained(v2a_ckpt_path).eval()
        
        if exists(t2a_ckpt_path):
            self.t2a_mapper = DiffusionPrior.from_pretrained(t2a_ckpt_path).eval()
        
        model_config = ckpt_path.rsplit('/', 2)[0] + '/config.yaml'
        with open(model_config, 'r') as f:
            self.config = yaml.safe_load(f)
        
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self.config['concat_y']:
            dit_model = DiT_models[self.config['model']]
        else:
            dit_model = DiT_cross_models[self.config['model']]
        dit = dit_model(
            input_size = (1024 // vae_scale_factor, 64 // vae_scale_factor),
            in_channels = 8,
            projection_class_embeddings_input_dim=512,
            caption_channels = 64,
            concat_y = self.config.get("concat_y", False),
            cross_class = self.config.get("cross_class", False),
        )

        # TODO: Get checkpoints
        def load_ckpt(model, ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt)
            print(f"Loaded checkpoint from {ckpt_path}")
            return model

        self.model = load_ckpt(DiTWrapper(dit, cond_speaker=True, concat_y=self.config.get("concat_y", False)), ckpt_path)
        self.model.eval()

        self.device = device
        self.vae.to(device)
        self.vocoder.to(device)
        self.clap_model.to(device)
        
        if exists(v2a_ckpt_path):
            self.clip_model.to(device)
            self.v2a_mapper.to(device)
        if exists(t2a_ckpt_path):
            self.t2a_mapper.to(device)
        self.model.to(device)

        # if exists(male_voice):
        #     wav, sr = torchaudio.load(male_voice)
        #     if sr != 16000:
        #         wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        #     self.male_voice = self.speech2spk_embed(wav.squeeze(0).to(device))
        
        # if exists(female_voice):
        #     wav, sr = torchaudio.load(female_voice)
        #     if sr != 16000:
        #         wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        #     self.female_voice = self.speech2spk_embed(wav.squeeze(0).to(device))
        
        if exists(male_voice):
            with open(male_voice, 'r') as f:
                self.male_voice = f.readlines()
        
        if exists(female_voice):
            with open(female_voice, 'r') as f:
                self.female_voice = f.readlines()

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor,
            self.vocoder.config.model_in_dim // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform

    @torch.no_grad()
    def __call__(
        self,
        modality,
        env_prompt,
        cont_prompt,
        style_prompt = None,
        speaker_audio = None,
        gender = None,
        batch_size = 1,
        num_inference_steps = 100,
        audio_length_in_s = 10,
        do_classifier_free_guidance = True,
        v2a_guidance_scale = None,
        guidance_scale = None,
        desc_guidance_scale = None,
        cont_guidance_scale = None,
        device=None,
        seed=None,
        progress=True,
        **kwargs,
    ):  

        if guidance_scale is None and desc_guidance_scale is None and cont_guidance_scale is None:
            do_classifier_free_guidance = False
        
        guidance = None
        if guidance_scale is None:
            guidance = "dual"
        else:
            guidance = "single"
        
        # description condition
        if modality == 'text':
            if do_classifier_free_guidance:
                if guidance == "dual":
                    env_prompt = [env_prompt] * 2 + [""] * 2
                if guidance == "single":
                    env_prompt = [env_prompt] + [""]
            else:
                env_prompt = [env_prompt]

            clap_inputs = self.clap_processor(
                text=env_prompt, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            c_desc = self.clap_model.get_text_features(**clap_inputs)
        
        elif modality == 'audio':
            audio_sample, sr = torchaudio.load(env_prompt)
            if sr != 48000:
                audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=48000)
            audio_sample = audio_sample[0]

            clap_inputs = self.clap_processor(audios=audio_sample, sampling_rate=48000, return_tensors="pt", padding=True).to(self.device)
            c_desc = self.clap_model.get_audio_features(**clap_inputs)

            if do_classifier_free_guidance:
                clap_inputs = self.clap_processor(text=[""], return_tensors="pt", padding=True).to(self.device)
                # uncond_embeds = self.clap_model.text_model(**clap_inputs).pooler_output
                # uc_desc = self.clap_model.text_projection(uncond_embeds)
                uc_desc = self.clap_model.get_text_features(**clap_inputs)
                
                if guidance == "dual":
                    c_desc = torch.cat((c_desc, c_desc, uc_desc, uc_desc))
                if guidance == "single":
                    c_desc = torch.cat((c_desc, uc_desc))
        
        elif modality == 'image':
            image_sample = Image.open(env_prompt)
            
            clip_inputs = self.clip_processor(images=image_sample, return_tensors="pt").to(self.device)
            # clip_embeds = self.clip_model.vision_model(**clip_inputs).pooler_output
            # clip_embeds = self.clip_model.visual_projection(clip_embeds)
            # clip_embeds = torch.nn.functional.normalize(clip_embeds, dim=-1)
            clip_embeds = self.clip_model.get_image_features(**clip_inputs)
            # clip_embeds = env_prompt
            text_cond = dict(text_embed = clip_embeds)

            c_desc = self.v2a_mapper.p_sample_loop(
                clip_embeds.shape,
                text_cond = text_cond,
                cond_scale = v2a_guidance_scale,
                timesteps = 100
            )

            c_desc = torch.nn.functional.normalize(c_desc, dim=-1)

            if do_classifier_free_guidance:
                clip_inputs = self.clap_processor(text=[""], return_tensors="pt", padding=True).to(self.device)
                uc_desc = self.clap_model.get_text_features(**clip_inputs)

                if guidance == "dual":
                    c_desc = torch.cat((c_desc, c_desc, uc_desc, uc_desc))
                if guidance == "single":
                    c_desc = torch.cat((c_desc, uc_desc))
        
        # speaker style conditon
        if style_prompt is not None:
            if do_classifier_free_guidance:
                if guidance == "dual":
                    style_prompt = [style_prompt] * 2 + [""] * 2
                if guidance == "single":
                    style_prompt = [style_prompt] + [""]
            else:
                style_prompt = [style_prompt]

            clap_text_tokens = self.clap_tokenizer(style_prompt, return_tensors="pt", padding=True).to(self.device)
            c_style = self.clap_model.get_text_features(**clap_text_tokens)
        
        elif gender is not None:
            
            if gender == 'man':
                spk_audio = random.choice(self.male_voice).strip()
            elif gender == 'woman':
                spk_audio = random.choice(self.female_voice).strip()
            else:
                spk_audio = random.choice(self.male_voice + self.female_voice).strip()
            spk_audio, sr = torchaudio.load(spk_audio)
            
            if sr != 16000:
                spk_audio = torchaudio.functional.resample(spk_audio, orig_freq=sr, new_freq=16000)
            if spk_audio.shape[0] > 1:
                spk_audio = spk_audio.mean(0, keepdim=True)
            c_style = self.speech2spk_embed(spk_audio.squeeze(0))
            
        elif speaker_audio is not None:
            spk_audio, sr = torchaudio.load(speaker_audio)
            if sr != 16000:
                spk_audio = torchaudio.functional.resample(spk_audio, orig_freq=sr, new_freq=16000)
            if spk_audio.shape[0] > 1:
                spk_audio = spk_audio.mean(0, keepdim=True)
            c_style=self.speech2spk_embed(spk_audio.squeeze(0))
        else:
            c_style = torch.zeros(192).to(self.device)
        
        cont_tokens = self.get_text(cont_prompt, add_blank=True)
        cont_tokens = cont_tokens.unsqueeze(0).to(self.device)
        cont_lengths = torch.LongTensor([cont_tokens.shape[-1]]).to(self.device)

        mu_y, y_mask = self.model.process_content(cont_tokens, cont_lengths, c_style)
        
        if self.model.concat_y:
            mu_y_vae = self.model.proj(mu_y.unsqueeze(1))
        else:
            mu_y_vae = mu_y.unsqueeze(1)

        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        height = int(audio_length_in_s * 1.024 / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        # prepare latent variables
        num_channels_latents = self.model.dit.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            c_desc.dtype,
            device=device,
            generator=torch.manual_seed(seed) if seed else None,
            latents=None,
        )
        
        # denoising loop

        if guidance == "dual":
            z = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
            cfg_scale = (desc_guidance_scale, cont_guidance_scale)
        elif guidance == "single":
            z = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            cfg_scale = guidance_scale
        
        if self.model.concat_y:
            mu_y_vae = torch.nn.functional.pad(mu_y_vae, (0, 0, 0, z.shape[-2] - mu_y_vae.shape[-2]), value=0)
        if do_classifier_free_guidance:
            if self.config['concat_y']:
                if guidance == "dual":
                    mu_y_vae = torch.cat([mu_y_vae, self.model.y_embedding.unsqueeze(0)] * 2, dim=0)
                elif guidance == "single":
                    mu_y_vae = torch.cat([mu_y_vae, self.model.y_embedding.unsqueeze(0)], dim=0)
            else:
                null_y = self.model.dit.y_embedder.y_embedding.unsqueeze(1).repeat(1, mu_y_vae.shape[-2], 1)
                if guidance == "dual":
                    mu_y_vae = torch.cat([mu_y_vae, null_y.unsqueeze(0)] * 2, dim=0)
                elif guidance == "single":
                    mu_y_vae = torch.cat([mu_y_vae, null_y.unsqueeze(0)], dim=0)
        
        model_kwargs = dict(
            y=mu_y_vae,
            mask=y_mask.long().squeeze(1) if not self.model.concat_y else None,
            class_labels=c_desc,
            guidance=guidance,
            cfg_scale = cfg_scale,
        )
        
        diffusion = create_diffusion(str(num_inference_steps))
        
        # Sample images:
        samples = diffusion.p_sample_loop(
            self.model.dit.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress,
            device=self.device
        )
        if guidance == "dual":
            samples = samples[: len(samples) // 4]
        elif guidance == "single":
            samples = samples[: len(samples) // 2]

        mel_spectrogram = self.decode_latents(samples)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        audio = self.normalize_wav(audio)

        return audio, mel_spectrogram
    

class EvalPipeline():
    def __init__(
        self,
        vae,
        clap_model: ClapModel,
        clap_processor: ClapProcessor,
        device,
        cmudict_path='voicedit/cmu_dictionary'
    ):
        self.vae = vae
        self.vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm-m-full", subfolder="vocoder").eval()
        self.clap_model = clap_model
        self.clap_processor = clap_processor
        self.speech2spk_embed = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_ecapa_wavlm_joint")
        self.device = device
        self.vocoder.to(device)
        self.cmudict = cmudict.CMUDict(cmudict_path)
    
    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor,
            self.vocoder.config.model_in_dim // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform

    @torch.autocast('cuda', dtype=torch.float16)
    @torch.no_grad()
    def __call__(
        self,
        model,
        clap_input_features,
        clap_is_longer,
        cont_prompt,
        spk_embeds,
        batch_size = 1,
        num_inference_steps = 100,
        audio_length_in_s = 10,
        do_classifier_free_guidance = True,
        guidance_scale = None,
        desc_guidance_scale = None,
        cont_guidance_scale = None,
        seed=None,
        concat_y = None,
        **kwargs,
    ):
        if guidance_scale is None and desc_guidance_scale is None and cont_guidance_scale is None:
            do_classifier_free_guidance = False
        
        guidance = None
        if guidance_scale is None:
            guidance = "dual"
        else:
            guidance = "single"
        
        # embeds = self.clap_model.audio_model(input_features=clap_input_features, is_longer=clap_is_longer).pooler_output
        # c_desc = self.clap_model.audio_projection(embeds)

        c_desc = self.clap_model.get_audio_features(input_features=clap_input_features, is_longer=clap_is_longer)

        if do_classifier_free_guidance:
            clap_inputs = self.clap_processor(text=[""], return_tensors="pt", padding=True).to(self.device)
            # uncond_embeds = self.clap_model.text_model(**clap_inputs).pooler_output
            # uc_desc = self.clap_model.text_projection(uncond_embeds)
            uc_desc = self.clap_model.get_text_features(**clap_inputs)
            
            if guidance == "dual":
                c_desc = torch.cat((c_desc, c_desc, uc_desc, uc_desc))
            if guidance == "single":
                c_desc = torch.cat((c_desc, uc_desc))
                
        # content condition
        # if do_classifier_free_guidance:
        #     if guidance == "dual":
        #         cont_prompt = ([cont_prompt] + ["_"]) * 2
        #     if guidance == "single":
        #         cont_prompt = [cont_prompt] + ["_"]

        # cont_tokens = self.text_processor(
        #     text=[cont_prompt], 
        #     padding=True,
        #     truncation=True,
        #     max_length=1000,
        #     return_tensors="pt"
        # ).to(self.device)
        # cont_embed_mask = cont_tokens.attention_mask
        
        cont_tokens = self.get_text(cont_prompt, add_blank=True)
        cont_tokens = cont_tokens.unsqueeze(0).to(self.device)
        cont_lengths = torch.LongTensor([cont_tokens.shape[-1]]).to(self.device)
        
        mu_y, y_mask = model.process_content(cont_tokens, cont_lengths, spk_embeds)
        mu_y_vae = mu_y.unsqueeze(1)
        if concat_y:
            mu_y_vae = model.proj(mu_y_vae)

        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        height = int(audio_length_in_s * 1.024 / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        # prepare latent variables
        num_channels_latents = model.dit.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            c_desc.dtype,
            device=self.device,
            generator=torch.manual_seed(seed) if seed else None,
            latents=None,
        )
        
        # denoising loop
        if guidance == "dual":
            z = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
            cfg_scale = (desc_guidance_scale, cont_guidance_scale)
        elif guidance == "single":
            z = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            cfg_scale = guidance_scale
        
        if concat_y:
            mu_y_vae = torch.nn.functional.pad(mu_y_vae, (0, 0, 0, z.shape[-2] - mu_y_vae.shape[-2]), value=0)
            if do_classifier_free_guidance:
                if guidance == "dual":
                    mu_y_vae = torch.cat([mu_y_vae, model.y_embedding.unsqueeze(0)] * 2, dim=0)
                elif guidance == "single":
                    mu_y_vae = torch.cat([mu_y_vae, model.y_embedding.unsqueeze(0)], dim=0)
            
            model_kwargs = dict(
                y=mu_y_vae,
                class_labels=c_desc,
                guidance=guidance,
                cfg_scale=cfg_scale,
            )
        
        else:
            null_y = model.dit.y_embedder.y_embedding.repeat(mu_y_vae.shape[-2], 1)
            null_y = null_y.unsqueeze(0).unsqueeze(0)
            if do_classifier_free_guidance:
                if guidance == "dual":
                    mu_y_vae = torch.cat([mu_y_vae, null_y] * 2, dim=0)
                elif guidance == "single":
                    mu_y_vae = torch.cat([mu_y_vae, null_y], dim=0)
            
            model_kwargs = dict(
                y=mu_y_vae,
                mask=y_mask.squeeze(1).bool(),
                class_labels=c_desc,
                guidance=guidance,
                cfg_scale=cfg_scale,
            )
        
        diffusion = create_diffusion(str(num_inference_steps))
        
        # Sample images
        samples = diffusion.p_sample_loop(
            model.dit.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            device=self.device
        )
        if guidance == "dual":
            samples = samples[: len(samples) // 4]
        if guidance == "single":
            samples = samples[: len(samples) // 2]
        
        mel_spectrogram = self.decode_latents(samples)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        audio = self.normalize_wav(audio)

        return mu_y[0].transpose(0,1), mel_spectrogram.transpose(2,3), audio