import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as L
import numpy as np
import cv2

class LitModel(L.LightningModule):
    def __init__(self, audio = True, n_stacked = 3):
        super().__init__()
        self.encoder = Encoder(audio = audio, n_stacked = n_stacked)


        # with open('data.txt', 'r') as f:
            # self.data = 


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                 std=[66.675, 63.84, 57.375]),
        ])

    def forward(self, hist):
        inp = []
        for im in hist:
            im = im.astype(np.float32)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            o = self.transform(im).float()
            inp.append(o)

        inp = torch.stack(inp).unsqueeze(0)
        z = self.encoder({
            'audio': None,
            'video': inp
        })
        return z

class Encoder(nn.Module):
    def __init__(self, audio = True, n_stacked = 3):
        super().__init__()
        pass
        
    def forward(self, x):
        # x['video'] is [batch_size, 3, 3, 224, 224]
        # x['audio'] is [batch_size, 1, 57, 160]
        pass