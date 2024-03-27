import os
import cv2
import json
import torch
import matplotlib.pyplot as plt
from vit_pytorch import SimpleViT
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import csv
import numpy as np

## create a train code for images
def compute_loss(xyz_gt, xyz_pred):
    loss = F.mse_loss(xyz_gt, xyz_pred)
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 11,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
).to(device)

# batch = 8
epochs = 100
optimizer = torch.optim.Adam(vit.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.9
)

# train = torch.randn(800, 3, 256, 256).to(device), torch.randn(800, 5).to(device)
# val = torch.randn(240, 3, 256, 256).to(device), torch.randn(240, 5).to(device)

class CustomDataset(Dataset):
    def __init__(self, data, runs, imgs_folder, audio_file, targets_file, transform=None, device=None):
        self.data = self.get_data(data, runs, imgs_folder)
        self.targets = self.get_targets(data, runs, targets_file)
        self.transform = transform

    def get_data(self, data, runs, imgs_folder):
        images = []
        for runs_folder in runs:
            imgs_folder = os.path.join(data, runs_folder, imgs_folder)
            for img in os.listdir(imgs_folder):
                img = os.path.join(imgs_folder, img)
                images.append(cv2.imread(img))
        return images

    def get_targets(self, data, runs, targets_file):
        targets_arr = []
        for runs_folder in runs:
            targets_file = os.path.join(data, runs_folder, targets_file)
            with open(targets_file, "r") as f:
                targets = json.load(f)
                for target in targets:
                    targets_arr.append(target)
        return targets_arr
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]

        if self.transform:
            sample = self.transform(sample)
            label = torch.tensor(label)

        return sample, label
    

data = "/home/punygod_admin/SoundSense/soundsense/data/playbyear_runs"

train_runs = []
with open('/home/punygod_admin/SoundSense/soundsense/baselines/mulsa/src/datasets/data/train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        train_runs.append(row[0])
print(train_runs)

test_runs = []
with open('/home/punygod_admin/SoundSense/soundsense/baselines/mulsa/src/datasets/data/val.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        test_runs.append(row[0])
print(test_runs)

imgs_folder = "video"
audio_file = "processed_audio.wav"
targets_file = "keyboard_teleop.json"

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

train_dataset = CustomDataset(data, train_runs, imgs_folder, audio_file, targets_file, transform=transform, device=device)
test_dataset = CustomDataset(data, test_runs, imgs_folder, audio_file, targets_file, transform=transform, device=device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# print(train[0].size(0), val[0].shape)
# inputs, xyzgt_gt = train[0], train[1]
# val_inputs, val_xyzgt_gt = val[0], val[1]
# print(len(train_loader), len(test_loader))
training_loss, validation_loss = [], []

for k in range(epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        # print(np.asarray(inputs).shape, np.asarray(targets).shape)
        xyzgt_pred = vit(inputs.to(device).float())
        loss = compute_loss(targets.to(device).float(), xyzgt_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"epoch {k}, loss {running_loss}")
    training_loss.append(running_loss)
    scheduler.step()

    val_running_loss = 0.0
    for val_inputs, val_targets in test_loader:
        xyzgt_pred = vit(val_inputs.to(device).float())
        loss = compute_loss(val_targets.to(device).float(), xyzgt_pred)
        val_running_loss += loss.item()

    print(f"epoch {k}, val loss {val_running_loss}")
    validation_loss.append(val_running_loss)

      
# save weights after training
torch.save(vit.state_dict(), "vit_weights.pth")

# plot the training and validation loss
plt.plot(training_loss, label="training loss")
plt.plot(validation_loss, label="validation loss")
plt.legend()
plt.savefig("loss.png")
