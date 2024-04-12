from functools import partial

import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_opt import Blip2OPT

from lavis.models.eva_gvt import EVAVisionTransformer

from apex.normalization import FusedLayerNorm


@registry.register_model("blip2_gvt")
class Blip2GVT(Blip2OPT):
    def init_vision_encoder(
        self, *_
    ):
        encoder = EVAVisionTransformer(img_size=224, patch_size=14, depth=24,
                                        mlp_ratio=2.6667, num_heads=16, embed_dim=1024,
                                        drop_path_rate=0, xattn=True,
                                        qkv_bias=True,
                                        norm_layer=partial(FusedLayerNorm, eps=1e-6),
                                        rope=True, pt_hw_seq_len=16, intp_freq=True,
                                        naiveswiglu=True, subln=True)
        self.patch_embed_dim = 1024
        ln_vision = nn.LayerNorm(self.patch_embed_dim)


        return encoder, ln_vision