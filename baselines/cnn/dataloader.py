import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import pickle
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path, audio = True, n_stack = 3):
        print("Loading pickle file", path)
        self.n_stack = n_stack
        self.data = pickle.load(open(path, 'rb'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                 std=[66.675, 63.84, 57.375]),
        ])
        self.audio = audio
        print("Loaded pickle file", path)
        print("Trajectory length", len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(len(self.data[idx]), len(self.data))
        audio_obs, obs, action = self.data[idx]
        if self.audio:
            audio_obs = (audio_obs.reshape(1, 57,160)).float()
        else:
            audio_obs = []
        final_obs = []
        for i in range(self.n_stack):
            o = np.transpose(obs[i], (1,2,0)).astype(np.float32)
            o = self.transform(o).float()
            # print(o)
            final_obs.append(o)

        final_obs = torch.stack(final_obs)  
        action = torch.tensor(action).float()

        return [audio_obs, final_obs, action]

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, use_audio = True):
        super().__init__()
        self.batch_size = batch_size
        self.audio = use_audio

    def setup(self, stage=None):
        # Define your dataset here
        if self.audio:
            path = '/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/data_audio.pkl'

        else:
            path = '/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/data.pkl'
        train_data = CustomDataset(path, audio = self.audio, n_stack= 10)
        self.train_dataset = train_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size)