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
        self.sequence_files = sorted(glob.glob(root_dir + "/metadata/*.txt"))

        self.total_sequence_length = 0
        self.final_sequences = []
        self.sequence_lengths = []
        self.datas = []
        for i, path in enumerate(self.sequence_files):
            with open(path, 'rb') as f:
                length = int(f.read())
            
            if length == 0:
                continue

            self.final_sequences.append(path)

            self.sequence_lengths.append(length)
            self.total_sequence_length += length
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
        # print("Getting", sequence_idx, frame_idx)
        # episode = self.datas[sequence_idx]
        
        audio_obs = torch.load(self.final_sequences[sequence_idx].replace('metadata', 'audio').replace('txt', 'pt'))
        obs = torch.load(self.final_sequences[sequence_idx].replace('metadata', 'video').replace('txt', 'pt'))
        actions = torch.load(self.final_sequences[sequence_idx].replace('metadata', 'actions').replace('txt', 'pt')).float()
        length = len(audio_obs)
        # episode = pkl.load(open(self.sequence_files[sequence_idx], "rb"))
        if frame_idx + self.sequence_length > length:
            frame_idx = length - self.sequence_length
        audio_obs = audio_obs[frame_idx:frame_idx+self.sequence_length]
        obs = obs[frame_idx:frame_idx+self.sequence_length]
        actions = actions[frame_idx:frame_idx+self.sequence_length]
        return audio_obs, obs, actions


# Create dataset instance
if __name__ == "__main__":
    # Example usage:
    root_dir = "/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/audio/"  # Path to the dataset folder
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        # transforms.Resize((224, 224)),  # Resize input images to the required size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize input images
    ])

    dataset = ActionDataset(root_dir, 10, transform)
    print(len(dataset))
    # Create data loader
    batch_size = 10
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over batches
    for data in data_loader:
        # inputs is a batch of input sequences (batch_size, seq_length, channels, height, width)
        # targets is a batch of target action sequences (batch_size, seq_length, num_actions)
        print("Audio shape:", data[0].shape)
        print("Image shape:", data[1].shape)
        print("Action shape:", data[2].shape)
        break  # Break after processing the first batch