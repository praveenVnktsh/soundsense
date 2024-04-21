from torchvision.models import resnet18
import torch

if __name__ == "__main__":
    # model = resnet18()
    model = torch.load("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/pretrained/H.pth.tar")
    print(model)