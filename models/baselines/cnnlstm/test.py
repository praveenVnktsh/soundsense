import torch
import torch.nn as nn

# Size: [batch_size, seq_len, input_size]
input = torch.randn(16, 10, 512)
lstm_hidden_dim=64
output_dim=11
lstm_layers=2
dropout=0.5
lstm = nn.LSTM(
        input_size=512, 
        hidden_size=lstm_hidden_dim, 
        num_layers=lstm_layers, 
        dropout=dropout, 
        batch_first=True
)

output, _ = lstm(input)
print(output.size())