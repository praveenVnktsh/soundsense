import torch
torch.set_num_threads(1)

import sys
sys.path.append("")
from models.baselines.mulsa.src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from models.baselines.mulsa.src.models import Actor
from torchvision import transforms
import pytorch_lightning as pl
import yaml
import configargparse

class MULSAInference(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # p = configargparse.ArgParser()
        # p.add("-c", "--config", is_config_file=True, default="models/baselines/mulsa/conf/imi/imi_learn.yaml")
        # p.add("--batch_size", default=32)
        # p.add("--lr", default=1e-4, type=float)
        # p.add("--gamma", default=0.9, type=float)
        # p.add("--period", default=3)
        # p.add("--epochs", default=100, type=int)
        # p.add("--resume", default=None)
        # p.add("--num_workers", default=8, type=int)
        # # imi_stuff
        # p.add("--conv_bottleneck", required=True, type=int)
        # p.add("--exp_name", required=True, type=str)
        # p.add("--encoder_dim", required=True, type=int)
        # p.add("--action_dim", default=3, type=int)
        # p.add("--num_stack", required=True, type=int)
        # p.add("--frameskip", required=True, type=int)
        # p.add("--use_mha", default=False, action="store_true") ## multi head attention
        # # data
        # p.add("--train_csv", default="src/datasets/data/train.csv")
        # p.add("--val_csv", default="src/datasets/data/val.csv")
        # p.add("--data_folder", default="../../data/mulsa/data")
        # p.add("--resized_height_v", required=True, type=int)
        # p.add("--resized_width_v", required=True, type=int)
        # p.add("--resized_height_t", required=True, type=int)
        # p.add("--resized_width_t", required=True, type=int)
        # p.add("--num_episode", default=None, type=int)
        # p.add("--crop_percent", required=True, type=float)
        # p.add("--ablation", required=True)
        # p.add("--num_heads", required=True, type=int)
        # p.add("--use_flow", default=False, action="store_true")
        # p.add("--task", type=str)
        # p.add("--norm_audio", default=False, action="store_true")
        # p.add("--aux_multiplier", type=float)
        # p.add("--nocrop", default=False, action="store_true")

        # args = p.parse_args()
        # args.batch_size *= torch.cuda.device_count()
        
        # v_encoder = make_vision_encoder(args.encoder_dim)
        # a_encoder = make_audio_encoder(args.encoder_dim * args.num_stack, args.norm_audio)

        # self.actor = Actor(v_encoder, a_encoder, args)
        config_path = 'conf/imi/test.yaml'
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        v_encoder = make_vision_encoder(config['encoder_dim'])
        a_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'])

        self.actor = Actor(v_encoder, a_encoder, config)

        # self.transform_cam = T.Compose(
        #     [
        #         T.Resize((self.resized_height_v, self.resized_width_v)),
        #         T.CenterCrop((self._crop_height_v, self._crop_width_v)),
        #     ]
        # )

        # cam_gripper_framestack = torch.stack(
        #         [
        #             self.transform_cam(
        #                 self.load_image(self.trial, "video", timestep)
        #             )
        #             for timestep in frame_idx
        #         ],
        #         dim=0,
        #     )
        self.transform_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['resized_height_v'], config['resized_width_v'])),
            transforms.ToTensor(),
        ])

    def forward(self, inp):

        video = inp['video'] #list of images
        audio = inp['audio']

        audio = torch.tensor(audio).unsqueeze(0)

        video = torch.stack([self.transform_image(img) for img in video], dim=0)
        video = video.unsqueeze(0)
        
        x = video, audio
        out = self.actor(x)
        # print(out.shape)
        return out