import torch
torch.set_num_threads(1)
import numpy as np
import sys
sys.path.append("")
from models.baselines.mulsa.src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from models.baselines.mulsa.src.models.imi_models import Actor
from torchvision import transforms
import pytorch_lightning as pl
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import configargparse
import os
class MULSAInference(pl.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as info:
            self.config = yaml.load(info.read(), Loader=yaml.FullLoader) 
        v_encoder = make_vision_encoder(self.config['encoder_dim'])
        a_encoder = make_audio_encoder(self.config['encoder_dim'] * self.config['num_stack'], self.config['norm_audio'])
        self.use_audio = "ag" in self.config["modalities"].split("_")
        self.weights = None
        self.prv_layer_output = None
        self.actor = Actor(v_encoder, a_encoder, self.config)
        self.loss_cce = torch.nn.CrossEntropyLoss(weight= torch.tensor([1]*8))
        self.num_stack = self.config['num_stack']
        print("NUM STACK: ", self.num_stack)
        # self.transform_image = A.Compose([
        #     A.Normalize(mean=0.485, std=0.229, max_pixel_value=1.0),
        #     ToTensorV2(),])
        # self.moddevi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform_image = A.Compose([           
                #  A.Normalize(mean=0.485,
                #                  std=0.229, max_pixel_value= 1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ]
        )

    def forward(self, inp):
        print(self.device)
        video = inp["video"] #list of images
        video = torch.stack([self.transform_image(image=img)['image'] for img in video], dim=0).to(self.device)
        video = video.unsqueeze(0)
        # print(video.shape)
        
        if self.use_audio:
            audio = inp["audio"].to(self.device).unsqueeze(0)
            x = video, audio
        else:
            x = video, None
        out = self.actor(x) # tuple of 3 tensors (main output, weights, prev layer output)
        self.weights = out[1]
        self.prv_layer_output = out[2]
        return out[0]
    
    def get_weights(self):
        return self.weights
    
    def get_prev_layer_output(self):
        return self.prv_layer_output
    
    def get_activations_gradient(self):
        return self.actor.get_activations_gradient()
    
    def get_activations(self):
        return self.actor.get_activations()