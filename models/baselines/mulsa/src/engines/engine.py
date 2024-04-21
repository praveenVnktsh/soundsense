import time
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt



class ImiEngine(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader, scheduler, config):
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config

        self.loss_cce = torch.nn.CrossEntropyLoss(
            reduction='mean',
            weight= torch.tensor(config['loss_weights']),
            # ignore_index=10
        )
        self.validation_step_outputs = []

        self.wrong = 1
        self.correct = 0
        self.total = 0
        
        self.output_sequence_length = config['output_sequence_length']
        self.action_dim = config["action_dim"]
        self.decoder_type = config["output_model"] if  'output_model' in config.keys() else 'simple'
        self.save_hyperparameters(config)

    def compute_loss(self, xyz_gt, xyz_pred):
        
        gt_future_actions, past_actions = xyz_gt # both are [batch_size, seq_len, action_dim]
        B, sl, _ = gt_future_actions.size()

        # xyz_pred = [B, Seq, Action_dim]

        # if self.decoder_type == 'simple':
        #     gt_future_actions = gt_future_actions.view(B, -1)
        #     pred_future_actions = xyz_pred.view(B, -1)
        #     loss = self.loss_cce(pred_future_actions, gt_future_actions)


        # if self.decoder_type in ["layered", 'multi_head', 'lstm']:    
        
        loss = 0
        for i in range(gt_future_actions.size(1)):
            loss += self.loss_cce(xyz_pred[:, i, :], gt_future_actions[:, i, :])
        # loss /= sl
        
        return loss

    def training_step(self, batch, batch_idx):
        # use idx in batch for debugging
        inputs, xyzgt_gt = batch
        # print("training_step input shape", inputs[0].size())
        xyzgt_pred, weights, _ = self.actor(inputs)  # , idx)
        loss = self.compute_loss(xyzgt_gt, xyzgt_pred)
        self.log_dict(
            {"train/loss": loss}, prog_bar=True, on_epoch=True
        )
        # action_pred = torch.argmax(action_logits, dim=1)
        # acc = (action_pred == demo).sum() / action_pred.numel()
        # self.log("train/acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # inputs, demo, xyzrpy_gt, start = batch
        # action_logits, xyzrpy_pred, weights = self.actor(inputs, start)  # , idx)
        # loss, immi_loss, aux_loss = self.compute_loss(
        #     demo, action_logits, xyzrpy_gt, xyzrpy_pred
        # )
        inputs, xyzgt_gt = batch
        xyzgt_pred, weights, _ = self.actor(inputs) 
        loss = self.compute_loss(xyzgt_gt, xyzgt_pred)
        self.log_dict(
            {"val/loss": loss}, prog_bar=True, on_epoch=True
        )
        # self.log_dict(
        #     {"val/immi_loss": immi_loss, "val/aux_loss": aux_loss}, prog_bar=True
        # )
        # action_pred = torch.argmax(action_logits, dim=1)
        # # number of corrects and total number
        # val_output = ((action_pred == demo).sum(), action_pred.numel())
        # self.validation_step_outputs.append(val_output)
        # # numerator = 0
        # # divider = 0
        # # for cor, total in val_output:
        # #     numerator += cor
        # #     divider += total
        # # acc = numerator / divider
        # # self.log("val/acc", acc, on_step=False, on_epoch=True)
        # return ((action_pred == demo).sum(), action_pred.numel())
        return loss


    # def on_validation_epoch_end(self):
    #     numerator = 0
    #     divider = 0
    #     for cor, total in self.validation_step_outputs:
    #         numerator += cor
    #         divider += total
    #     acc = numerator / divider
    #     self.log("val/acc", acc, on_step=False, on_epoch=True)

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader


    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
