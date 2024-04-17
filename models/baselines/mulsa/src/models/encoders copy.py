import os
import numpy as np
import soundfile as sf
from torchvision.models import resnet18
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
import torch
from torch import nn
import torchvision
from transformers import AutoProcessor, ASTModel, ASTConfig
# from perceiver_pytorch import Perceiver
import torch.nn.functional as F
import torchaudio


class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""

    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_dim is not None:
            self.fc = nn.Linear(512, out_dim)
        self.vision_gradients = None
        self.vision_activations = None

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        
        # h_v = x.register_hook(self.vision_activation_hook)
        self.vision_activations = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x

    def vision_activation_hook(self, grad):
        self.vision_gradients = grad

class Spec_Encoder(Encoder):
    def __init__(self, feature_extractor, out_dim=None, norm_audio=False):
        super().__init__(feature_extractor, out_dim)
        self.norm_audio = norm_audio
        sr = 16000
        # self.mel = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        # )
        self.audio_gradients = None
        self.audio_activations = None

    def forward(self, log_spec):
        # EPS = 1e-8
        EPS = 1
        # print("waveform ",(waveform.size()))
        # spec = self.mel(waveform.float())
        # log_spec = torch.log(spec + EPS)
        # assert log_spec.size(-2) == 64
        # if self.norm_audio:
        #     log_spec /= log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100]
        x = super().forward(log_spec)
        self.audio_activations = x

        # h_a = x.register_hook(self.audio_activation_hook)
        return x
    
    def audio_activation_hook(self, grad):
        self.audio_gradients = grad

class ASTEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # self.config = ASTConfig()
        self.sr = 16000 # Hardcoded
        self.fc = None
        if out_dim is not None:
            self.fc = nn.Linear(1214*768, out_dim) # Adds 1.5B params

    def forward(self, audio):
        # audio file is decoded on the fly
        inputs = self.processor(audio.cpu().numpy().squeeze(1), sampling_rate=self.sr, return_tensors="pt").to(self.model.device)
        with torch.no_grad(): # finetune all layers?
            outputs = self.model(**inputs)
        x = outputs.last_hidden_state
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x
    

class HubertEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("ntu-spml/distilhubert")
        self.model = ASTModel.from_pretrained("ntu-spml/distilhubert")
        # self.config = ASTConfig()
        self.sr = 16000 # Hardcoded
        self.fc = None
        if out_dim is not None:
            self.fc = nn.Linear(1214*768, out_dim) # Adds 1.5B params

    def forward(self, audio):
        # audio file is decoded on the fly
        inputs = self.processor(audio.cpu().numpy().squeeze(1), sampling_rate=self.sr, return_tensors="pt").to(self.model.device)
        with torch.no_grad(): # finetune all layers?
            outputs = self.model(**inputs)
        x = outputs.last_hidden_state
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x

def make_vision_encoder(out_dim=None):
    vision_extractor = resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
    print(vision_extractor)
    
    vision_extractor.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    # return Vision_Encoder(vision_extractor, out_dim)
    return Encoder(vision_extractor, out_dim)


# def make_flow_encoder():
#     input_dim = 2 * 10 * 14
#     encoder = nn.Sequential([]
#         nn.Flatten(1),
#         nn.Linear(input_dim, 2048),
#         nn.Linear(2048, 1024),
#         nn.Linear(1024, 512),
#     )
#     return encoder

def make_audio_encoder(out_dim=None, norm_audio=False, model="spec"):
    if model == "spec":
        audio_extractor = resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
        audio_extractor.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        audio_extractor = create_feature_extractor(audio_extractor, ["layer4.1.relu_1"])
        return Spec_Encoder(audio_extractor, out_dim, norm_audio)
    elif model == "ast":
        return ASTEncoder(out_dim)


if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_vision_encoder(64, 1280)
    print(encoder(inp).shape)
    # episode_folder = "/home/punygod_admin/SoundSense/soundsense/data/mulsa/data/30"
    # audio_encoder = make_audio_encoder(256*6, model="ast")
    # audio_gripper1 = sf.read(os.path.join(episode_folder, "processed_audio.wav"))[0]
    #     # print("Audio loaded")

    # audio_gripper = [
    #     x for x in audio_gripper1 if x is not None
    # ]
    # audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
    # audio_gripper = (audio_gripper).reshape(1,-1)
    # audio_clip = clip_resample(audio_gripper, 0, 3*48000)
    # print(audio_clip.shape)
    # output = audio_encoder({"audio": {"array": audio_clip}})
    # print(output.shape)

