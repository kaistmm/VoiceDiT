import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from .audio import get_mel_from_wav, raw_waveform_to_fbank, TacotronSTFT
from text import text_to_sequence, cmudict
from text.symbols import symbols
from .utils import intersperse


class AudioDataset(Dataset):
    def __init__(self, args, df, df_noise, clap_processor, rir_path=None, cmudict_path='voicedit/cmu_dictionary', add_blank=True, train=True):
        self.df = df
        self.df_noise = df_noise
        
        self.paths = args.paths
        self.noise_paths = args.noise_paths

        self.uncond_text_prob = args.uncond_text_prob
        self.add_noise_prob = args.add_noise_prob
        self.reverb_prob = args.reverb_prob
        
        self.duration = 10
        self.target_length = int(self.duration * 102.4)
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

        self.clap_processor = clap_processor
        self.train = train
        if not train:
            self.noise_dict = dict()
        
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank

    def get_mel(self, audio, _stft):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, _, _ = _stft.mel_spectrogram(audio)
        return torch.squeeze(melspec, 0).float()
    
    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm
    
    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)
        rir, fs = torchaudio.load(rir_file)
        rir = rir.to(dtype=audio.dtype)
        rir = rir / torch.linalg.vector_norm(rir, ord=2, dim=-1)

        return torchaudio.functional.fftconvolve(audio, rir)[:,:self.target_length * 160]
        
    def pad_wav(self, wav, target_len, random_cut=False):
        n_channels, wav_len = wav.shape
        if n_channels == 2:
            wav = wav.mean(-2, keepdim=True)

        if wav_len > target_len:
            if random_cut:
                i = random.randint(0, wav_len - target_len)
                return wav[:, i:i+target_len]
            return wav[:, :target_len]
        elif wav_len < target_len:
            wav = F.pad(wav, (0, target_len-wav_len))
        return wav

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = os.path.join(self.paths[row.data], row.file_path)
        
        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        wav_len = waveform.shape[1] 
        if wav_len > self.target_length * 160:
            wav_len = self.target_length * 160
        
        fbank_clean = self.get_mel(waveform[0, :wav_len], self.stft)
        
        waveform = self.pad_wav(waveform, self.target_length * 160)
        
        if row.data in ['as_speech_en']:
            spk_emb=torch.load(file_path.replace('/dataset/', '/spk_emb/').replace('.wav', '.pt'), map_location=torch.device('cpu'))

        if row.data in ['cv', 'voxceleb', 'libri'] and len(self.df_noise) > 0:
            if random.random() < self.reverb_prob and hasattr(self, 'rir_files'):
                waveform = self.reverberate(waveform)

            if random.random() < self.add_noise_prob:
                noise_row = self.df_noise.iloc[random.randint(0, len(self.df_noise)-1)]
                noise, sr = torchaudio.load(os.path.join(self.noise_paths[noise_row.data], noise_row.file_path))
                noise = torchaudio.functional.resample(noise, orig_freq=sr, new_freq=16000)
                noise = self.pad_wav(noise, self.target_length * 160, random_cut=True)
                if torch.linalg.vector_norm(noise, ord=2, dim=-1).item() != 0.0:
                    snr = torch.Tensor(1).uniform_(2, 10)
                    waveform = torchaudio.functional.add_noise(waveform, noise, snr)
            # else:
            #     if index not in self.noise_dict:
            #         noise_idx = random.randint(0, len(self.df_noise)-1)
            #         noise_row = self.df_noise.iloc[noise_idx]
            #         noise, sr = torchaudio.load(os.path.join(self.noise_paths[noise_row.data], noise_row.file_path))
            #         noise = torchaudio.functional.resample(noise, orig_freq=sr, new_freq=16000)
            #         noise = self.pad_wav(noise, self.target_length * 160, random_cut=True)
            #         if torch.linalg.vector_norm(noise, ord=2, dim=-1).item() == 0.0:
            #             noise = torch.zeros_like(noise)
            #         snr = torch.Tensor(1).uniform_(4, 20)
            #         self.noise_dict[index] = (noise, snr)
            #     else:
            #         noise, snr = self.noise_dict[index]
            #     waveform = torchaudio.functional.add_noise(waveform, noise, snr)

            spk_emb=torch.load(file_path.replace('/LibriTTS_R_16k/', '/LibriTTS_R_spk/').replace('.wav', '.pt'), map_location=torch.device('cpu'))

        if type(spk_emb) != torch.Tensor:
            spk_emb = spk_emb[0]

        fbank, _, waveform = raw_waveform_to_fbank(
            waveform[0], 
            target_length=self.target_length, 
            fn_STFT=self.stft
        )

        text = row.text
        tokenized_text = self.get_text(text, add_blank=self.add_blank)

        # resample to 48k for clap
        wav_48k = torchaudio.functional.resample(waveform, orig_freq=16000, new_freq=48000)
        clap_inputs = self.clap_processor(audios=wav_48k, return_tensors="pt", sampling_rate=48000)
        
        return text, tokenized_text, fbank, spk_emb, clap_inputs, self.normalize_wav(waveform), fbank_clean
        
    def __len__(self):
        return len(self.df)


class CollateFn(object):

    def __call__(self, examples):
        B = len(examples)

        fbank = torch.stack([example[2] for example in examples])
        spk_embs = torch.cat([example[3] for example in examples])
        clap_input_features = torch.cat([example[4].input_features for example in examples])
        clap_is_longer = torch.cat([example[4].is_longer for example in examples])
        audios = [example[5] for example in examples]

        y_max_length = max([example[6].shape[-1] for example in examples])
        x_max_length = max([example[1].shape[-1] for example in examples])
        n_feats = examples[0][6].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, example in enumerate(examples):
            x_, y_ = example[1], example[6]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
        
        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)

        return {
            "audio": audios,
            "fbank": fbank, 
            "spk": spk_embs, 
            "text": x,
            "text_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "clap_input_features": clap_input_features,
            "clap_is_longer": clap_is_longer,
            "texts": [example[0] for example in examples],
        }