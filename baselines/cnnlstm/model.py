import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMWithResNetForRobot(nn.Module):
    def __init__(self, hidden_dim, output_dim, lstm_layers, dropout):
        super(CNNLSTMWithResNetForRobot, self).__init__()
        
        # Load pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=resnet.fc.out_features, hidden_size=hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        
        # Define fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Extract features using ResNet
        with torch.no_grad():  # Disable gradient computation for the ResNet backbone
            features = self.resnet(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(features.view(features.size(0), -1, features.size(1)))
        
        # Take the output from the last time step
        lstm_last_output = lstm_out[:, -1, :]
        
        # Apply fully connected layer for output
        out = self.fc(lstm_last_output)
        
        return out

# Example usage:
hidden_dim = 64  # Dimensionality of the LSTM hidden state
output_dim = 11  # Output dimensionality (number of actions)
lstm_layers = 2  # Number of LSTM layers
dropout = 0.5  # Dropout probability

# Create model instance
model = CNNLSTMWithResNetForRobot(hidden_dim, output_dim, lstm_layers, dropout)
print(model)
