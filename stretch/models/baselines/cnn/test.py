import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as L
from model import Encoder


if __name__ == '__main__':
    use_audio = False
    model = LitModel(audio = use_audio)

    input = torch.randn(1, 3, 64, 64)