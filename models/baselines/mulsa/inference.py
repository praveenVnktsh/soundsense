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
        self.actor = Actor(v_encoder, a_encoder, self.config)
        self.loss_cce = torch.nn.CrossEntropyLoss(weight= torch.tensor([1]*11))
        # self.transform_image = transforms.Compose([
        #     # transforms.ToPILImage(), ## why do need this transformation?
        #     # transforms.Resize((self.config['resized_width_v'], self.co    nfig['resized_height_v'])),
        #     # transforms.Resize((self.config['resized_height_v'], self.config['resized_width_v'])),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=0.485,
        #             std=0.229,),
        # ])
        self.transform_image = A.Compose([
            A.Normalize(mean=0.485, std=0.229, max_pixel_value=1.0),
            ToTensorV2(),])
        # self.transform_image = A.Compose([                A.Normalize(mean=0.485,
        #                          std=0.229, max_pixel_value= 1.0),
        #         # A.Normalize(mean=[0.485, 0.456, 0.406],
        #         #                  std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
        #         ToTensorV2(),
        #     ], additional_targets= {
        #         f'image{i}': 'image' for i in range(self.num_stack)})
        # self.transform_image = transforms.Compose([
        #     transforms.ToPILImage(), ## why do need this transformation?
        #     transforms.Resize((self.config['resized_height_v'], self.config['resized_width_v'])),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225], ),
        # ])
        
        # self.transform_cam = A.Compose([
        #         A.Resize(height=self.resized_height_v, width=self.resized_width_v),
        #         A.Normalize(
        #             # mean=0.5,
        #             # std=0.5,
                    # mean=[0.485, 0.456, 0.406],
                    # std=[0.229, 0.224, 0.225], 
        #             max_pixel_value= 1.0
        #         ),
        #         ToTensorV2(),
        #     ])

    def forward(self, inp):

        video = inp["video"] #list of images
        # print(type(video[0]))
        video = torch.stack([self.transform_image(image=img)['image'] for img in video], dim=0)
        video = video.unsqueeze(0)
        
        if self.use_audio:
            audio = inp["audio"]
            x = video, audio
        else:
            x = video, None
        print(x[0].shape)
        out = self.actor(x) # tuple of 3 tensors (main output, weights, prev layer output)
        return out[0]
    
    def get_activations_gradient(self):
        return self.actor.get_activations_gradient()
    
    def get_activations(self):
        return self.actor.get_activations()