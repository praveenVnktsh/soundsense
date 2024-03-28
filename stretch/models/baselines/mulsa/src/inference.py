import torch
torch.set_num_threads(1)

import sys
sys.path.append("")
from models.baselines.mulsa.src.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from models.baselines.mulsa.src.imi_models import Actor

import pytorch_lightning as pl

import configargparse

class MULSAInference(pl.LightningModule):
    def __init__(self):
        super().__init__()

        p = configargparse.ArgParser()
        p.add("-c", "--config", is_config_file=True, default="models/baselines/mulsa/conf/imi/imi_learn.yaml")
        p.add("--batch_size", default=32)
        p.add("--lr", default=1e-4, type=float)
        p.add("--gamma", default=0.9, type=float)
        p.add("--period", default=3)
        p.add("--epochs", default=100, type=int)
        p.add("--resume", default=None)
        p.add("--num_workers", default=8, type=int)
        # imi_stuff
        p.add("--conv_bottleneck", required=True, type=int)
        p.add("--exp_name", required=True, type=str)
        p.add("--encoder_dim", required=True, type=int)
        p.add("--action_dim", default=3, type=int)
        p.add("--num_stack", required=True, type=int)
        p.add("--frameskip", required=True, type=int)
        p.add("--use_mha", default=False, action="store_true") ## multi head attention
        # data
        p.add("--train_csv", default="src/datasets/data/train.csv")
        p.add("--val_csv", default="src/datasets/data/val.csv")
        p.add("--data_folder", default="../../data/mulsa/data")
        p.add("--resized_height_v", required=True, type=int)
        p.add("--resized_width_v", required=True, type=int)
        p.add("--resized_height_t", required=True, type=int)
        p.add("--resized_width_t", required=True, type=int)
        p.add("--num_episode", default=None, type=int)
        p.add("--crop_percent", required=True, type=float)
        p.add("--ablation", required=True)
        p.add("--num_heads", required=True, type=int)
        p.add("--use_flow", default=False, action="store_true")
        p.add("--task", type=str)
        p.add("--norm_audio", default=False, action="store_true")
        p.add("--aux_multiplier", type=float)
        p.add("--nocrop", default=False, action="store_true")

        args = p.parse_args()
        args.batch_size *= torch.cuda.device_count()
        
        v_encoder = make_vision_encoder(args.encoder_dim)
        a_encoder = make_audio_encoder(args.encoder_dim * args.num_stack, args.norm_audio)

        self.actor = Actor(v_encoder, a_encoder, args)

    def forward(self, x):
        return self.actor(x)
    
