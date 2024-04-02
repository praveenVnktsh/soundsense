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
import json

class LitModel(L.LightningModule):
    def __init__(self, action_path):
        super().__init__()
        
        with open(action_path, 'rb') as f:
            self.actions = json.load(f)
        self.counter = 0
    def forward(self, hist):
        act = torch.tensor(self.actions[self.counter])
        
        self.counter += 1
        if self.counter >= len(self.actions):
            act = torch.zeros((11))
            act[-1] = 1

        
        return act
        

class Encoder(nn.Module):
    def __init__(self, audio = True, n_stacked = 3):
        super().__init__()
        pass
        
    def forward(self, x):
        # x['video'] is [batch_size, 3, 3, 224, 224]
        # x['audio'] is [batch_size, 1, 57, 160]
        pass