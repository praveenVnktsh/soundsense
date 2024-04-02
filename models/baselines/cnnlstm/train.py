import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import CNNLSTMWithResNetForActionPrediction

import sys
sys.path.append('../mulsa/src/datasets/')
from imi_datasets import ImitationEpisode


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

    train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, 1, num_workers=config["num_workers"], shuffle=False)

    # Initialize Lightning Trainer
    trainer = pl.Trainer(max_epochs=config["max_epochs"], accelerator="auto")

    # Initialize model
    model = CNNLSTMWithResNetForActionPrediction(
        sequence_length=config["num_stack"], 
        lstm_hidden_dim=config["lstm_hidden_dim"], 
        output_dim=config["output_dim"], 
        lstm_layers=config["lstm_layers"], 
        dropout=config["dropout"], 
        audio=config["audio"]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main(config_path = 'train.yaml')
