import torch
torch.set_num_threads(1)
import yaml
import sys
import os
sys.path.append(f'{os.environ["SOUNDSENSE_ROOT"]}/models')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

from imi_datasets_dagger import ImitationEpisode
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
torch.set_float32_matmul_precision('medium')

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)

def main(config):

    # dataset_root
    
        
    np.random.seed(0)
    run_ids = os.listdir(config['dataset_root'])
    np.random.permutation(run_ids)
    split = int(config['train_val_split']*len(run_ids))
    train_episodes = run_ids[:split]
    val_episodes = run_ids[split:]
    print("Train episodes: ", len(train_episodes))
    print("Val episodes: ", len(val_episodes))
    config['train_episodes'] = train_episodes
    config['val_episodes'] = val_episodes
    print(train_episodes)

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

    train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, config["batch_size"], num_workers=config["num_workers"], shuffle=False)

    v_encoder = make_vision_encoder(config['encoder_dim'])
    a_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'], model=config["audio_encoder"])
    # a_encoder = make_audio_encoder(config['encoder_dim'] * config['num_stack'], config['norm_audio'])

    imi_model = Actor(v_encoder, a_encoder, config)
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['period'], gamma=config['gamma']
    )

    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, scheduler, config
    )
    start_training(config, config['log_dir'], pl_module)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Fri-04-26-09:54sorting_imi_vg_ag_simple_seqlen_1_mha_spec_audio_len_5_num_stacks_6", help="Path to config file")
    parser.add_argument("--dagger_run_id", type=int, default=0, help="Dagger run id")    

    root = '/home/punygod_admin/SoundSense/soundsense/models/basqelines/mulsa/lightning_logs'
    args = parser.parse_args()
    with open(f'/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/dagger.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['exp_name'] = f'dagger_run_{args.dagger_run_id}_{args.base_model}'
    config.update(vars(args))

    main(config)
