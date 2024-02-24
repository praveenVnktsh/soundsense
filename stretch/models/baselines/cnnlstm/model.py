import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):

    def __init__(self,):
        super().__init__()

        # self.nn = nn.Sequential(
        #     # nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 2, padding = 0),
        #     # nn.MaxPool2d()
        #     # nn.Linear
        # )
        self.nn = None

    def forward(self, inp):
        return np.random.rand(5, 1)
        
    