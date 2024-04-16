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
        self.decoder_type = config["decoder_type"] 
        self.action_dim = config["action_dim"]
        self.output_sequence_length = config['output_sequence_length']
    
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        
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

        
        if self.decoder_type == 'simple':
            print("Creating simple decoder")
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
            print("Creating layered decoder")
            self.decoder = [
                nn.Sequential(nn.Linear(self.action_dim, self.action_dim), nn.Tanh()).to(self.device)
                for i in range(config["output_sequence_length"])
            ]
            self.decoder = nn.ModuleList(self.decoder)
        if self.decoder_type == 'lstm':
            print("Creating LSTM decoder")
            self.lstm_hidden_layers = config['lstm_hidden_layers']
            self.decoder = nn.LSTM(
                self.action_dim, 
                self.action_dim, 
                num_layers=self.lstm_hidden_layers, 
                batch_first=True
            )
            # self.input_past_actions_dim = config["input_past_actions_dim"]
            # self.history_encoder_dim = config["history_encoder_dim"]
            # self.history_mlp = [nn.Sequential(nn.Linear(self.input_past_actions_dim*self.action_dim, self.history_encoder_dim), nn.Tanh()).to(self.device)] + \
            #     [nn.Sequential(nn.Linear(self.layernorm_embed_shape+self.history_encoder_dim, self.layernorm_embed_shape), nn.Tanh()).to(self.device)]
            
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
        print("Total parameters:", count_parameters(self), "Million")
        
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
            out = out.view(-1, self.output_sequence_length, self.action_dim)
        else:
            pred = self.decoder_bottleneck(mlp_inp).to(self.device) # the bottleneck
            if self.decoder_type == "layered":
                out = torch.tensor([]).to(self.device)
                for i in range(len(self.decoder)):
                    pred = self.decoder[i](pred)
                    out = torch.cat((out, pred.unsqueeze(1)), dim=1)

            elif self.decoder_type == "multi_head":
                out = torch.tensor([]).to(self.device)
                
                for i in range(len(self.decoder)):
                    # outs.append(self.decoder[i](pred))
                    out = torch.cat((out, pred.unsqueeze(1)), dim=1)
                # out = torch.stack(outs, dim=1) # [batch, stack_future_actions_dim, action_dim]

            elif self.decoder_type == "lstm":
                batch_size = mlp_inp.shape[0]
                h0 = torch.zeros(self.lstm_hidden_layers, batch_size, self.action_dim).to(self.device)
                c0 = torch.zeros(self.lstm_hidden_layers, batch_size, self.action_dim).to(self.device)
                out = torch.tensor([]).to(self.device)
                pred = pred.unsqueeze(1)
                for t in range(self.output_sequence_length):
                    pred, (h0, c0) = self.decoder(pred, (h0, c0))  
                    out = torch.cat((out, pred), dim=1)

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
    # layered, multi_head, lstm, simple
    config={
        'decoder_type': 'simple',
        'use_mha': False,
        'encoder_dim': 256,
        'num_stack': 6,
        'num_heads': 8,
        'modalities': 'vg_ag',
        'action_dim': 11,
        'output_sequence_length': 30,
        'audio_len' : 3,
        'grayscale': False,
        'norm_audio': True,
        'audio_encoder': 'spec'
    }
    torch.manual_seed(0)
    vision_encoder = make_vision_encoder(config['encoder_dim'])
    # audio_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'])
    audio_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'], model=config["audio_encoder"])
    model = Actor(vision_encoder, audio_encoder, config).cuda()
    audio_in = torch.zeros((2, 1, 64, 301)).cuda()
    video_in = torch.zeros((2, 6, 3, 75, 100)).cuda()
    out, weights, mlp_inp = model([video_in, audio_in])
    print(out.shape)
    # summary(model, [(3, 75, 100), (1, 64, 301)])
    # print(model)
    # for i in range(5):
    #     print(out[0][i])
