import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class CNNLSTMWithResNetForActionPrediction(pl.LightningModule):
    def __init__(self, sequence_length, lstm_hidden_dim, output_dim, lstm_layers, dropout, audio=False):
        super(CNNLSTMWithResNetForActionPrediction, self).__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  

        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_dim, 
            num_layers=lstm_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.num_history = sequence_length
        self.audio = audio

        # Define fully connected layer for output
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        
    def forward(self, x):
        # Extract features using ResNet
        x = x.view(-1, *x.shape[2:])  # [N_batch * N_history, C, H, W]

        # Extract features using ResNet
        with torch.no_grad():  # Disable gradient computation for the ResNet backbone
            features = self.resnet(x)

        # Apply LSTM
        features = features.view(features.size(0) // self.num_history, self.num_history, features.size(1))  # [N_batch, N_history, features_size]
        lstm_out, _ = self.lstm(features)
        # print(lstm_out.shape)
        # Apply fully connected layer for output
        out = self.fc(lstm_out)
        return out

    def training_step(self, batch, batch_idx):
        
        (obs, audio), actions = batch
        
        # print("train", obs.shape, actions.shape)

        outputs = self(obs)
        loss = nn.functional.cross_entropy(outputs, actions)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        (obs, audio), actions = batch
        # print("val", obs.shape, actions.shape)

        outputs = self(obs)
        loss = nn.functional.cross_entropy(outputs, actions)
        self.log('val_loss', loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Example usage:

# model = CNNLSTMWithResNetForActionPrediction(lstm_hidden_dim=64, output_dim=11, lstm_layers=2, dropout=0.5)
