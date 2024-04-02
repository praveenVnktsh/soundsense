import torch
torch.set_num_threads(1)

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
import configargparse

class MULSAInference(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config_path = '/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/imi/test.yaml'
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        v_encoder = make_vision_encoder(self.config['encoder_dim'])
        a_encoder = make_audio_encoder(self.config['encoder_dim'] * self.config['num_stack'], self.config['norm_audio'])

        self.actor = Actor(v_encoder, a_encoder, self.config)
        self.transform_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['resized_height_v'], self.config['resized_width_v'])),
            transforms.ToTensor(),
        ])
        

    def forward(self, inp):

        video = inp["video"] #list of images
        # video = torch.stack([self.transform_image(img) for img in video], dim=0)
        video = video.unsqueeze(0)
        # print(video.shape)

        if "ag" in self.config["modalities"].split("_"):
            audio = inp["audio"]
            audio = torch.tensor(audio).unsqueeze(0)
            x = video.cuda(), audio.cuda()
        else:
            x = video.cuda(), None

        out = self.actor(x) # tuple of 3 tensors (main output, weights, prev layer output)
        # print(out.shape)
        return out[0]
    
    def get_activations_gradient(self):
        return self.actor.get_activations_gradient()
    
    def get_activations(self):
        return self.actor.get_activations()