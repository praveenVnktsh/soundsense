import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import CNNLSTMWithResNetForActionPrediction
from models.baselines.mulsa.src.datasets.imi_datasets import ImitationEpisode


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

    # Create datasets
    # n_history = 10
    # train_dataset = ActionDataset(root_dir=root_dir, sequence_length=n_history,transform=transform, audio=False)
    # val_dataset = ActionDataset(root_dir="Dataset/val", transform=transform)

    # Create data loaders
    # batch_size = 16
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, 1, num_workers=config["num_workers"], shuffle=False)

    # Initialize Lightning Trainer
    trainer = pl.Trainer(max_epochs=100,)  # Adjust max_epochs and gpus as needed

    # Initialize model
    model = CNNLSTMWithResNetForActionPrediction(sequence_length=n_history, lstm_hidden_dim=64, output_dim=11, lstm_layers=2, dropout=0.5, audio=False)

    # Start training
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main(config_path = 'train.yaml')
