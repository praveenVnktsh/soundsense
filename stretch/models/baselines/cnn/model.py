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
class MelSpectrogramNet(nn.Module):
    def __init__(self):
        super(MelSpectrogramNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 10, 512)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 14 * 10)
        
        # Fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))
        
        return x



class LitModel(L.LightningModule):
    def __init__(self, audio = True, n_stacked = 3):
        super().__init__()
        self.encoder = Encoder(audio = audio, n_stacked = n_stacked)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                 std=[66.675, 63.84, 57.375]),
        ])

    def training_step(self, batch, batch_idx):
        
        audio_obs, obs, action, = batch
        # print(obs.shape)
        z = self.encoder({
            'audio': audio_obs,
            'video': obs
        })
        loss = F.cross_entropy(z, action)
        # print(loss)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
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
        



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class Encoder(nn.Module):
    def __init__(self, audio = True, n_stacked = 3):
        super().__init__()
        
        # video features
        self.video_backbone = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).children())[:-1])

        # audio features - ResNet 18
        # input size = 1, 57, 160
        self.audio = audio
        if audio:
            self.audio_backbone = MelSpectrogramNet()

        self.n_stack = n_stacked
        n_stacked = n_stacked + 1 if audio else n_stacked
        self.output_head = nn.Sequential(
            nn.Linear(512 * (n_stacked), 256 * n_stacked),
            nn.ReLU(),
            nn.Linear(256 * n_stacked, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
            # nn.Softmax(dim=1)
        )   

    def forward(self, x):
        # x['video'] is [batch_size, 3, 3, 224, 224]
        # x['audio'] is [batch_size, 1, 57, 160]
        features = []
        if self.audio:
            audio_features = self.audio_backbone(x['audio'])
            audio_features = audio_features.view(-1, 512)
            features.append(audio_features)
        
        # print(len(x['video']))s
        for i in range(self.n_stack):
            video_features = self.video_backbone(x['video'][:, i])
            video_features = video_features.view(-1, 512)
            features.append(video_features)

        
        features = torch.cat(features, dim=1)
        # print(features.shape)

        output = self.output_head(features)
        return output