from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn

# from engines.imi_engine import Future_Prediction
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

task2actiondim = {"pouring": 2, "insertion": 3}


class Actor(torch.nn.Module):
    def __init__(self, v_encoder, a_encoder, args):
        super().__init__()
        self.v_encoder = v_encoder
        self.a_encoder = a_encoder
        self.mlp = None
        self.layernorm_embed_shape = args.encoder_dim * args.num_stack
        self.encoder_dim = args.encoder_dim
        self.ablation = args.ablation
        self.use_vision = False
        self.use_audio = False
        self.use_mha = args.use_mha
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))

        ## load models
        self.modalities = self.ablation.split("_")
        print(f"Using modalities: {self.modalities}")
        # print(self.layernorm_embed_shape, len(self.modalities))
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.mha = MultiheadAttention(self.layernorm_embed_shape, args.num_heads)
        # print(self.embed_dim, self.layernorm_embed_shape)
        self.bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        )  # if we dont use mha

        # action_dim = 3 ** task2actiondim[args.task]

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3**args.action_dim),
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, 5) #6

    def forward(self, inputs):
        """
        Args:
            cam_gripper_framestack,audio_clip_g,
            vg_inp: [batch, num_stack, 3, H, W]
            a_inp: [batch, 1, T]

        """

        vg_inp, audio_g, = inputs
        embeds = []

        if "vg" in self.modalities:
            batch, num_stack, _, Hv, Wv = vg_inp.shape
            vg_inp = vg_inp.view(batch * num_stack, 3, Hv, Wv)
            vg_embeds = self.v_encoder(vg_inp)  # [batch * num_stack, encoder_dim]
            vg_embeds = vg_embeds.view(
                -1, self.layernorm_embed_shape
            )  # [batch, encoder_dim * num_stack]
            embeds.append(vg_embeds)
        if "ag" in self.modalities:
            batch, _, _ = audio_g.shape
            ag_embeds = self.a_encoder(audio_g)
            ag_embeds = ag_embeds.view(-1, self.layernorm_embed_shape)
            embeds.append(ag_embeds)

        if self.use_mha:
            mlp_inp = torch.stack(embeds, dim=0)  # [3, batch, D]
            # batch first=False, (L, N, E)
            # query = self.query.repeat(1, batch, 1) # [1, 1, D] -> [1, batch, D]
            # change back to 3*3
            mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)  # [1, batch, D]
            mha_out += mlp_inp
            mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
            # mlp_inp = mha_out.squeeze(0) # [batch, D]
        else:
            mlp_inp = torch.cat(embeds, dim=-1)
            # print(mlp_inp.shape)
            mlp_inp = self.bottleneck(mlp_inp)
            weights = None

        # action_logits = self.mlp(mlp_inp)
        xyzgt = self.aux_mlp(mlp_inp)
        # return action_logits, xyzrpy, weights
        return xyzgt, weights


if __name__ == "__main__":
    pass
    # vision_encoder = make_vision_encoder(128)
    # empty_input = torch.zeros((1, 3, 64, 101))
    # print(vision_encoder(empty_input).shape)
