import json
from pydantic import BaseModel
from typing import Optional
from dalle2_pytorch.train_configs import DiffusionPriorTrainConfig, TrackerConfig
from dalle2_pytorch.dalle2_pytorch import DiffusionPrior
from v2a_mapper.v2a_mapper import DiffusionPriorNetwork


def exists(val):
    return val is not None


class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    max_text_len: Optional[int] = None
    num_timesteps: Optional[int] = None
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_in: bool = False
    norm_out: bool = True
    attn_dropout: float = 0.
    ff_dropout: float = 0.
    final_proj: bool = True
    normformer: bool = False
    rotary_emb: bool = True

    class Config:
        extra = "allow"

    def create(self):
        kwargs = self.dict()
        return DiffusionPriorNetwork(**kwargs)


class DiffusionPriorConfig(BaseModel):
    clip = None
    net: DiffusionPriorNetworkConfig
    image_embed_dim: int
    image_size: int
    image_channels: int = 3
    timesteps: int = 1000
    sample_timesteps: Optional[int] = None
    cond_drop_prob: float = 0.
    loss_type: str = 'l2'
    predict_x_start: bool = True
    beta_schedule: str = 'cosine'
    condition_on_text_encodings: bool = True

    class Config:
        extra = "allow"

    def create(self):
        kwargs = self.dict()

        has_clip = exists(kwargs.pop('clip'))
        kwargs.pop('net')

        clip = None
        if has_clip:
            clip = self.clip.create()

        diffusion_prior_network = self.net.create()
        return DiffusionPrior(net = diffusion_prior_network, clip = clip, **kwargs)


class DiffusionPriorDataConfig(BaseModel):
    image_url: str                   # path to embeddings folder
    txt_url: str
    test_image_url: str
    test_txt_url: str
    split: float         # define train, validation, test splits for your dataset
    batch_size: int                  # per-gpu batch size used to train the model
    num_data_points: int = 25e7      # total number of datapoints to train on
    eval_every_seconds: int = 3600   # validation statistics will be performed this often


class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig
    data: DiffusionPriorDataConfig
    train: DiffusionPriorTrainConfig
    tracker: TrackerConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)