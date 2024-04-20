import torch
torch.set_num_threads(1)
import yaml
import sys
import os
sys.path.append(f'{os.environ["SOUNDSENSE_ROOT"]}/models')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

from imi_datasets import ImitationEpisode
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
    # train_episodes = run_ids
    # val_episodes = run_ids
    # train_episodes = run_ids[:3]
    # val_episodes = run_ids[:1]

    print("Train episodes: ", len(train_episodes))
    print("Val episodes: ", len(val_episodes))

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
    
    # save config
    # exp_dir = save_config(config)

    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, scheduler, config
    )
    # pl_module = ImiEngine(
    #     imi_model, optimizer, train_loader, train_loader, scheduler, config
    # )
    start_training(config, config['log_dir'], pl_module)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml", help="Path to config file")
    parser.add_argument("--mha", action='store_true', help="Use MHA")
    parser.add_argument("--decoder", type=str, help="Decoder type", 
                        choices=['layered', 'multi_head', 'lstm', 'simple'], required=True)
    parser.add_argument("--use_audio", action='store_true', help="Use audio")
    parser.add_argument('--output_sequence_length', type=int, required=True, help='Output sequence length')
    parser.add_argument('--audio_len', type=int, help='Output sequence length')
    parser.add_argument('--audio_encoder', type=str, default='spec', help='Audio encoder', choices=['spec', 'ast', 'hubert'])
    
    args = parser.parse_args()
    

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['use_mha'] = args.mha
    config['decoder_type'] = args.decoder
    config['output_sequence_length'] = args.output_sequence_length
    if args.audio_len:
        config['audio_len'] = args.audio_len
    if args.use_audio:
        config['modalities'] = 'vg_ag'
    else:
        config['modalities'] = 'vg'
    
    config['audio_encoder'] = args.audio_encoder
    config['exp_name'] = 'imi_' + config['modalities'] + '_' + config['decoder_type'] + "_seqlen_" + str(config['output_sequence_length'])
    if config['use_mha']:
        config['exp_name'] += '_mha'
    config['exp_name'] += "_"+config['audio_encoder']

    main(config)
