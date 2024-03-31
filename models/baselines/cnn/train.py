import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from dataloader import CustomDataModule
import pytorch_lightning as L
from model import LitModel


if __name__ == '__main__':
    use_audio = False
    model = LitModel(audio = use_audio, n_stacked = 10)
    train_loader = CustomDataModule(4, use_audio = use_audio)
    trainer = L.Trainer(max_epochs=1000)
    
    trainer.fit(model=model, train_dataloaders=train_loader)