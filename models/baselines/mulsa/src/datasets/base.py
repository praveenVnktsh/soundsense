import json
import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
import torchaudio
import soundfile as sf


class EpisodeDataset(Dataset):
    def __init__(self, data_folder="data"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        

    
    


    # @staticmethod
    # def load_flow(trial, stream, timestep):
    #     """
    #     Args:
    #         trial: the folder of the current episode
    #         stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
    #             for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
    #         timestep: the timestep of frame you want to extract
    #     """
    #     img_path = os.path.join(trial, stream, str(timestep) + ".pt")
    #     image = torch.as_tensor(torch.load(img_path))
    #     return image
