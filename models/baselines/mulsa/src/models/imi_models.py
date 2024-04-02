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


class Actor(torch.nn.Module):
    def __init__(self, v_encoder, a_encoder, config):
        super().__init__()
        self.v_encoder = v_encoder
        self.a_encoder = a_encoder

        self.layernorm_embed_shape = config["encoder_dim"] * config["num_stack"]
        self.encoder_dim = config["encoder_dim"]
        self.use_mha = config["use_mha"]
        self.modalities = config["modalities"].split("_")
    
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.mha = MultiheadAttention(self.layernorm_embed_shape, config["num_heads"])

        self.bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        )  # if we dont use mha

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(self.layernorm_embed_shape, 1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 3**config["action_dim"]),
        # )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, config["action_dim"]) #6

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
            mlp_inp = torch.stack(embeds, dim=0)  # [2, batch, D]
            # batch first=False, (L, N, E)
            # query = self.query.repeat(1, batch, 1) # [1, 1, D] -> [1, batch, D]
            # change back to 3*3
            mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)  # [1, batch, D]
            # print(weights.shape)
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
        return xyzgt, weights, mlp_inp
    
    def get_activations_gradient(self):
        return self.v_encoder.vision_gradients, self.a_encoder.audio_gradients

    def get_activations(self):
        if "ag" in self.modalities:
            return self.v_encoder.vision_activations.detach(), self.a_encoder.audio_activations.detach()
        else:
            return self.v_encoder.vision_activations.detach(), None
    
if __name__ == "__main__":
    pass
    # vision_encoder = make_vision_encoder(128)
    # empty_input = torch.zeros((1, 3, 64, 101))
    # print(vision_encoder(empty_input).shape)
