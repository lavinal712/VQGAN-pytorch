from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, Decoder
from .quantize import VectorQuantizer


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        kwargs = {
            "double_z": False,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "num_res_blocks": 2,
            "attn_resolutions": [16],
        }
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p, **kwargs)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p, **kwargs)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {"VQ-16": VQ_16, "VQ-8": VQ_8}
