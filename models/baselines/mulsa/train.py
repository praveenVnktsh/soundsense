import torch
torch.set_num_threads(1)
import yaml
from src.datasets.imi_datasets import ImitationEpisode
from src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from src.models.imi_models import Actor
from src.engines.engine import ImiEngine
from torch.utils.data import DataLoader
from src.train_utils import  start_training

import numpy as np
import random
import os
torch.multiprocessing.set_sharing_strategy("file_system")
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
print(sys.version)
print(torch.__version__)
print(torch.version.cuda)

def main(config_path):

    # dataset_root
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    np.random.seed(0)
    run_ids = os.listdir(config['dataset_root'])
    np.random.permutation(run_ids)
    
    split = int(config['train_val_split']*len(run_ids))
    train_episodes = run_ids[:split]
    val_episodes = run_ids[split:]

    train_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(config, run_id)
            for run_id in train_episodes
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(config, run_id, train=False)
            for run_id in val_episodes
        ]
    )

    # TODO: num_workers
    train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, 1, num_workers=config["num_workers"], shuffle=False)

    v_encoder = make_vision_encoder(config['encoder_dim'])
    a_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'])

    imi_model = Actor(v_encoder, a_encoder, config)
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['period'], gamma=config['gamma']
    )
    # save config
    # exp_dir = save_config(config)

    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, scheduler, config
    )
    start_training(config, config['log_dir'], pl_module)


if __name__ == "__main__":
    # import configargparse

    # p = configargparse.ArgParser()
    # p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
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
    # p.add("--use_mha", default=True, action="store_true") ## multi head attention
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
    main(config_path = 'conf/imi/train.yaml')
