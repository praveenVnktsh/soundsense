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
        print("Layernorm embed shape:", self.layernorm_embed_shape)
        self.encoder_dim = config["encoder_dim"]
        self.use_mha = config["use_mha"]
        self.modalities = config["modalities"].split("_")
        self.output_model = config["output_model"] if  'output_model' in config.keys() else 'aux'
        self.input_past_actions = config["input_past_actions"] if 'input_past_actions' in config.keys() else False
    
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        print("embed_dim:", self.layernorm_embed_shape )
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
        #     torch.nn.Linear(1024, 3**3),
        # )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.action_dim = config["action_dim"]
        self.aux_mlp = nn.Linear(self.layernorm_embed_shape, self.action_dim) #6

        if self.output_model == "seq_pred":
            self.seq_pred_mlp = nn.Linear(self.layernorm_embed_shape, config["stack_future_actions_dim"]*self.action_dim) 
        
        if self.output_model == "layered":
            self.layered_mlps = [nn.Sequential(nn.Linear(self.layernorm_embed_shape, self.action_dim), nn.Tanh()).to(self.device)] + \
                [nn.Sequential(nn.Linear(self.action_dim, self.action_dim), nn.Tanh()).to(self.device) for i in range(config["stack_future_actions_dim"])] 

        if self.output_model == "multi_head":
            self.multi_head_mlps = [nn.Sequential(nn.Linear(self.layernorm_embed_shape, self.action_dim), nn.ReLU()).to(self.device)] + \
                [nn.Linear(self.action_dim, self.action_dim).to(self.device) for i in range(config["stack_future_actions_dim"])] 

        # if self.output_model == 'lstm':
        #     self.output_sequence_length = config["output_sequence_length"] # we want atleast 3 seconds so we can have 30 subsequent actions
        #     self.decoder = nn.LSTM(self.layernorm_embed_shape, self.action_dim, num_layers=1, batch_first=True)

        if self.input_past_actions:  
            self.input_past_actions_dim = config["input_past_actions_dim"]
            self.history_encoder_dim = config["history_encoder_dim"]
            self.history_mlp = [nn.Sequential(nn.Linear(self.input_past_actions_dim*self.action_dim, self.history_encoder_dim), nn.Tanh()).to(self.device)] + \
                [nn.Sequential(nn.Linear(self.layernorm_embed_shape+self.history_encoder_dim, self.layernorm_embed_shape), nn.Tanh()).to(self.device)]

    def forward(self, inputs):
        """
        Args:
            cam_gripper_framestack,audio_clip_g,
            vg_inp: [batch, num_stack, 3, H, W]
            a_inp: [batch, 1, T]

        """
        if len(inputs) == 3:
            vg_inp, audio_g, history = inputs
        else:
            vg_inp, audio_g = inputs
            history = None
        embeds = []

        if "vg" in self.modalities:
            batch, num_stack, _, Hv, Wv = vg_inp.shape
            # print(vg_inp.shape,batch * num_stack, 3, Hv, Wv )
            vg_inp = vg_inp.view(batch * num_stack, 3, Hv, Wv)
            # print(vg_inp.dtype, vg_inp.shape)
            vg_embeds = self.v_encoder(vg_inp)  # [batch * num_stack, encoder_dim]
            vg_embeds = vg_embeds.view(
                -1, self.layernorm_embed_shape
            )  # [batch, encoder_dim * num_stack]
            embeds.append(vg_embeds)
        if "ag" in self.modalities:
            # batch, _, _ = audio_g.shape
            ag_embeds = self.a_encoder(audio_g)
            ag_embeds = ag_embeds.view(-1, self.layernorm_embed_shape)
            embeds.append(ag_embeds)
        
        if self.use_mha:
            mlp_inp = torch.stack(embeds, dim=0)  # [2, batch, D]
            # batch first=False, (L, N, E)
            # query = self.query.repeat(1, batch, 1) # [1, 1, D] -> [1, batch, D]
            # change back to 3*3
            mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp, average_attn_weights=False)  # [1, batch, D]
            # print("weights inside model:",weights.shape, weights)
            # weights.shape(1,8,1,1)
            mha_out += mlp_inp
            mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            # print("mha_out", mha_out.shape, "mlp_inp:", mlp_inp.shape)
            mlp_inp = self.bottleneck(mlp_inp)
            # mlp_inp = mha_out.squeeze(0) # [batch, D]
        else:
            mlp_inp = torch.cat(embeds, dim=-1)
            # print(mlp_inp.shape)
            mlp_inp = self.bottleneck(mlp_inp)
            weights = None


        if self.input_past_actions:
            # print("history", history.shape)
            history = history.view(-1, self.input_past_actions_dim*self.action_dim)
            history = self.history_mlp[0](history)
            mlp_inp = torch.concat([mlp_inp, history], dim=-1)
            mlp_inp = self.history_mlp[1](mlp_inp)
            # print("mlp_inp", mlp_inp.shape)

        # action_logits = self.mlp(mlp_inp)
        if self.output_model == "seq_pred":
            out = self.seq_pred_mlp(mlp_inp)

        elif self.output_model == "layered":
            out1 = self.layered_mlps[0](mlp_inp) # embed_dim, action_dim
            out = torch.Tensor([]).to(self.device)
            for i in range(1, len(self.layered_mlps)):
                out1 = self.layered_mlps[i](out1)
                out = torch.cat([out, out1.clone()], dim=0) # [Linear(action_dim, action_dim), Tanh()] (except last layer)
            out = out.view(-1, len(self.layered_mlps)-1, self.action_dim) # [batch, stack_future_actions_dim, action_dim]

        elif self.output_model == "multi_head":
            out1 = self.multi_head_mlps[0](mlp_inp) # embed_dim, action_dim
            out = torch.Tensor([]).to(self.device)
            for i in range(1, len(self.multi_head_mlps)):
                out2 = self.multi_head_mlps[i](out1) # all heads output on same out1
                out = torch.cat([out, out2.clone()], dim=0) # [Linear(action_dim, action_dim), ReLU()] (except last layer)
            out = out.view(-1, len(self.multi_head_mlps)-1, self.action_dim) # [batch, stack_future_actions_dim, action_dim]
            
        elif self.output_model == "aux":
            out = self.aux_mlp(mlp_inp)

        else:
            raise ValueError("Invalid output model in config")
        return out, weights, mlp_inp

    
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
