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
        self.decoder_type = config["output_model"] if  'output_model' in config.keys() else 'simple'
        self.input_past_actions = config["input_past_actions"] if 'input_past_actions' in config.keys() else False
    
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        
        print("Layernorm embed shape:", self.layernorm_embed_shape)
        print("embed_dim:", self.layernorm_embed_shape )
        
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.encoder_bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        ) 

        if self.use_mha:
            self.mha = MultiheadAttention(self.layernorm_embed_shape, config["num_heads"])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.action_dim = config["action_dim"]
        self.output_sequence_length = config['output_sequence_length']

        
        
        if self.decoder_type == 'simple':
            self.decoder = nn.Sequential(
                nn.Linear(self.layernorm_embed_shape, self.layernorm_embed_shape//2),
                nn.ReLU(),
                nn.Linear(self.layernorm_embed_shape//2, self.layernorm_embed_shape//2),
                nn.ReLU(),
                nn.Linear(self.layernorm_embed_shape//2, self.output_sequence_length * self.action_dim)
            )
        else:
            self.decoder_bottleneck = nn.Sequential(
                nn.Linear(self.layernorm_embed_shape, self.layernorm_embed_shape//2),
                nn.ReLU(),
                nn.Linear(self.layernorm_embed_shape//2, self.layernorm_embed_shape//4),
                nn.ReLU(),
                nn.Linear(self.layernorm_embed_shape//4, self.action_dim),
                nn.ReLU()
            )

        if self.decoder_type == "layered" or self.decoder_type == "multi_head":
            self.decoder = [
                nn.Sequential(nn.Linear(self.action_dim, self.action_dim), nn.Tanh()).to(self.device) 
                for i in range(config["stack_future_actions_dim"])
            ]
        if self.decoder_type == 'lstm':
            self.decoder = nn.LSTM(
                self.action_dim, 
                self.action_dim, 
                num_layers=1, 
                batch_first=True
            )
            # self.input_past_actions_dim = config["input_past_actions_dim"]
            # self.history_encoder_dim = config["history_encoder_dim"]
            # self.history_mlp = [nn.Sequential(nn.Linear(self.input_past_actions_dim*self.action_dim, self.history_encoder_dim), nn.Tanh()).to(self.device)] + \
            #     [nn.Sequential(nn.Linear(self.layernorm_embed_shape+self.history_encoder_dim, self.layernorm_embed_shape), nn.Tanh()).to(self.device)]

    def forward(self, inputs):
        """
        Args:
            cam_gripper_framestack,audio_clip_g,
            vg_inp: [batch, num_stack, 3, H, W]
            a_inp: [batch, 1, T]
        """
        vg_inp, audio_g = inputs
        
        # encoders
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
            mlp_inp = self.encoder_bottleneck(mlp_inp)
            # mlp_inp = mha_out.squeeze(0) # [batch, D]
        else:
            mlp_inp = torch.cat(embeds, dim=-1)
            # print(mlp_inp.shape)
            mlp_inp = self.encoder_bottleneck(mlp_inp)
            weights = None

        # apply layernorm.
        mlp_inp = self.layernorm(mlp_inp)

        if self.decoder_type == "simple":
            out = self.decoder(mlp_inp)
        else:
            mlp_inp = self.decoder_bottleneck(mlp_inp) # the bottleneck
            pred = self.decoder_bottleneck(mlp_inp) # the bottleneck

            if self.decoder_type == "layered":
                out = torch.tensor([]).to(self.device)
                outs = []
                for i in range(len(self.decoder)):
                    pred = self.decoder[i](pred)
                    outs.append(pred)
                out = torch.stack(outs, dim=1) # [batch, stack_future_actions_dim, action_dim]

            elif self.decoder_type == "multi_head":
                
                out = torch.tensor([]).to(self.device)
                outs = []
                for i in range(len(self.decoder)):
                    outs.append(self.decoder[i](pred))
                out = torch.stack(outs, dim=1) # [batch, stack_future_actions_dim, action_dim]

            elif self.decoder_type == "lstm":
                batch_size = mlp_inp.shape[0]
                h0 = torch.zeros(1, batch_size, self.action_dim).to(self.device)
                c0 = torch.zeros(1, batch_size, self.action_dim).to(self.device)
                outs = []
                for t in range(self.output_sequence_length):
                    out, (h0, c0) = self.decoder(pred, (h0, c0))  
                    outs.append(out)
                    pred = out.unsqueeze(1)

                out = torch.stack(outs, dim=1) # [batch, seq_len, action_dim]

        return out, weights, mlp_inp

    
    def get_activations_gradient(self):
        return self.v_encoder.vision_gradients, self.a_encoder.audio_gradients

    def get_activations(self):
        if "ag" in self.modalities:
            return self.v_encoder.vision_activations.detach(), self.a_encoder.audio_activations.detach()
        else:
            return self.v_encoder.vision_activations.detach(), None
    
if __name__ == "__main__":
    from encoders import make_vision_encoder, make_audio_encoder
    vision_encoder = make_vision_encoder(128)
    audio_encoder = make_audio_encoder(128)
    model = Actor(vision_encoder, audio_encoder, config={})
    # vision_encoder = make_vision_encoder(128)
    # empty_input = torch.zeros((1, 3, 64, 101))
    # print(vision_encoder(empty_input).shape)
