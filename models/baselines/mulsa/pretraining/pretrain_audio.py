from torchvision.models import resnet18
import torch
import torchvision
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import torchaudio.transforms as T
import torchaudio
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt

class ESC50Dataset(Dataset):
    def __init__(self, metadata_file, data_path, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_name = self.metadata.iloc[idx]['filename']
        file_path = os.path.join(self.data_path, file_name)
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Apply transformations if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        # Get label
        label = self.metadata.iloc[idx]['target']
        
        return waveform, label



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define path to metadata file and audio files
    metadata_file = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretraining/ESC-50-master/meta/esc50.csv"
    data_path = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretraining/ESC-50-master/audio"

    # Define transformations (you may need to adjust these)
    transform = torchvision.transforms.Compose([
        T.MelSpectrogram(sample_rate=44100),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=100)
    ])

    # Create dataset instance
    dataset = ESC50Dataset(metadata_file, data_path, transform=transform)


    # Load ResNet18 model
    model = resnet18()

    # Load state dictionary
    # state_dict = torch.load("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretrained/H.pth.tar")

    # # Transfer relevant weights manually
    # model_state_dict = model.state_dict()
    # for name, param in state_dict.items():
    #     if name in model_state_dict:
    #         model_state_dict[name].copy_(param)

    # # Load modified state dictionary into ResNet18 model
    # model.load_state_dict(model_state_dict)
    model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
    model.fc = torch.nn.Linear(512, 50)
    model.to(device)
    print(model)


    # Train test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create train and test dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train model using train_dataloader
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 200
    history = []
    for epoch in range(n_epochs):
        for i, (waveform, label) in tqdm(enumerate(train_dataloader)):
            waveform = waveform.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(waveform)
            loss = criterion(output, label)
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    print("Training complete")

    # Plot loss curve
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretrained/loss.png")



    # Save model weights
    torch.save(model.state_dict(), "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretrained/esc_pretrained.pth")

    # Evaluate accuracy on test dataloader
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveform, label in test_dataloader:
            waveform = waveform.to(device)
            label = label.to(device)
            output = model(waveform)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

