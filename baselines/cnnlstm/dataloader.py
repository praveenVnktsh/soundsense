import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import pickle as pkl
class ActionDataset(Dataset):
    def __init__(self, root_dir, sequence_length = 10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_files = sorted(glob.glob(root_dir + "/*.pkl"))

        self.total_sequence_length = 0
        self.sequence_lengths = []
        for i, path in enumerate(self.sequence_files):
            data = pkl.load(open(path, "rb"))
            print("Loaded", path, len(data))
            self.sequence_lengths.append(len(data))
            self.total_sequence_length += len(data)
            if i > 1:
                break
        print("Dataset has a total of", self.total_sequence_length, "frames")
        self.sequence_length = sequence_length
        
    def __len__(self):
        return self.total_sequence_length
    
    def get_sequence_index(self, idx):
        for i, length in enumerate(self.sequence_lengths):
            if idx < length:
                return i, idx
            idx -= length
        return None, None

    def __getitem__(self, idx):
        sequence_idx, frame_idx = self.get_sequence_index(idx)
        print("Getting", sequence_idx, frame_idx)
        episode = pkl.load(open(self.sequence_files[sequence_idx], "rb"))
        if frame_idx + self.sequence_length > len(episode):
            frame_idx = len(episode) - self.sequence_length

        relevant_sequence = episode[frame_idx:frame_idx+self.sequence_length]
        audio_obs = [x[0] for x in relevant_sequence]
        obs = [x[1] for x in relevant_sequence]
        actions = [x[2] for x in relevant_sequence]

        audio_obs = torch.tensor(audio_obs)
        obs = self.transform(obs)
        actions = torch.tensor(actions)

        return audio_obs, obs, actions

# Example usage:
root_dir = "/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/audio"  # Path to the dataset folder
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize input images to the required size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize input images
])

# Create dataset instance
dataset = ActionDataset(root_dir, 10, transform)

# Create data loader
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches
for inputs, targets in data_loader:
    # inputs is a batch of input sequences (batch_size, seq_length, channels, height, width)
    # targets is a batch of target action sequences (batch_size, seq_length, num_actions)
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    break  # Break after processing the first batch
