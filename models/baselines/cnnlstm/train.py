import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from model import CNNLSTMWithResNetForActionPrediction
from dataloader import ActionDataset

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_dir = root_dir = "/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/hundred/"  # Path to the dataset folder
# Create datasets
n_history = 10
train_dataset = ActionDataset(root_dir=root_dir, sequence_length=n_history,transform=transform, audio=False)
# val_dataset = ActionDataset(root_dir="Dataset/val", transform=transform)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Initialize Lightning Trainer
trainer = pl.Trainer(max_epochs=100,)  # Adjust max_epochs and gpus as needed

# Initialize model
model = CNNLSTMWithResNetForActionPrediction(sequence_length=n_history, lstm_hidden_dim=64, output_dim=11, lstm_layers=2, dropout=0.5, audio=False)

# Start training
trainer.fit(model, train_loader)
