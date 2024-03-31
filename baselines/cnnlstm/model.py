import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class CNNLSTMWithResNetForActionPrediction(pl.LightningModule):
    def __init__(self, sequence_length, lstm_hidden_dim, output_dim, lstm_layers, dropout):
        super(CNNLSTMWithResNetForActionPrediction, self).__init__()
        self.save_hyperparameters()
        # Load pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
        # Define LSTM layerwant to keep any logs or checkpoints, and there doesn't appear to be an obvious way to do that.
        # print(resnet.fc.out_features)
        # print(lstm_hidden_dim)
        # print(lstm_layers)
        # print(dropout)


        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_dim, 
            num_layers=lstm_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.num_history = sequence_length

        # Define fully connected layer for output
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        
    def forward(self, x):
        # Extract features using ResNet
        x = x.view(-1, *x.shape[2:])  # [N_batch * N_history, C, H, W]
        # Extract features using ResNet
        with torch.no_grad():  # Disable gradient computation for the ResNet backbone
            features = self.resnet(x)
        # Apply LSTM
        features = features.view(x.size(0) // self.num_history, self.num_history, -1)  # [N_batch, N_history, features_size]
        # print(features.shape)
        lstm_out, _ = self.lstm(features)
        # print(lstm_out.shape)
        # Apply fully connected layer for output
        out = self.fc(lstm_out)
        return out

    def training_step(self, batch, batch_idx):
        audio_obs, obs, actions = batch
        outputs = self(obs)
        loss = nn.functional.cross_entropy(outputs, actions)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Example usage:
# model = CNNLSTMWithResNetForActionPrediction(lstm_hidden_dim=64, output_dim=11, lstm_layers=2, dropout=0.5)
