import torch
torch.set_num_threads(1)

from src.datasets.imi_datasets import ImitationEpisode
from src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from src.models.imi_models import Actor
from src.engines.engine import ImiEngine
from torch.utils.data import DataLoader
from src.train_utils import save_config, start_training
import pandas as pd
import numpy as np
import random
import os
torch.multiprocessing.set_sharing_strategy("file_system")
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
def strip_sd(state_dict, prefix):
    """
    strip prefix from state dictionary
    """
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def main(args):
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    ## write code to split train test from total TODO    
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode

    train_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(args, i, args.data_folder)
            for i in range(train_num_episode)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(args, i, args.data_folder, False)
            for i in range(val_num_episode)
        ]
    )
    
    # create weighted sampler to balance samples
    train_label = []
    ## TODO
    # print("hi", len(train_set.datasets))
    ### Not using for now 
    '''
    for episode in train_set.datasets:
        # print(len(episode))
    #     for idx in range(len(episode)):
    #         train_label.append(episode.get_demo(idx))
        print(len(episode))
        for i in range(len(episode)):
            train_label.append(random.randint(0, 2))
    # print(np.array(train_label).shape)
    # histogram of label
            
    # TODO: Is this weighted sampler used to rectify the class imbalance?
    class_sample_count = np.zeros(pow(3, args.action_dim))
    for t in np.unique(train_label):
        class_sample_count[t] = len(np.where(train_label == t)[0])
    weight = 1.0 / (class_sample_count + 1e-5)
    samples_weight = np.array([weight[t] for t in train_label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )
    '''

    # TODO: num_workers
    train_loader = DataLoader(
        train_set, args.batch_size, num_workers=args.num_workers)#, sampler=sampler
    # )
    val_loader = DataLoader(val_set, 1, num_workers=args.num_workers, shuffle=False)

    # v encoder
    v_encoder = make_vision_encoder(args.encoder_dim)
    # a encoder
    a_encoder = make_audio_encoder(args.encoder_dim * args.num_stack, args.norm_audio)

    imi_model = Actor(v_encoder, a_encoder, args)#.cuda()
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.period, gamma=args.gamma
    )
    # save config
    exp_dir = save_config(args)
    # pl stuff
    # print(train_loader[0])
    # for batch, (d, label) in enumerate(train_loader):
    #     print(d.shape, label.shape)
    # for batch, (d1, label1) in enumerate(val_loader):
    #     print(d1.shape, label.shape)
    # for batch_idx, sample in enumerate(train_loader):
    #     print(sample, len(sample))
    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, scheduler, args
    )
    start_training(args, exp_dir, pl_module)


if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_learn.yaml")
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
    p.add("--data_folder", default="../../data/playbyear_runs")
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
    main(args)
