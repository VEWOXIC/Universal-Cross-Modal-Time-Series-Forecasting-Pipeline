from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate

import numpy as np
import torch
import torch.nn as nn

import os
import time
import warnings

import json

from tqdm import tqdm

warnings.filterwarnings('ignore')

class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)

    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args, is_LLM=True)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Get the data for training, validation, or testing.
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type='loader')
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type='loader')
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type='loader')

        return data_loader

    def _forward_step(self, iter):
        """
        Forward step for the model.
        """
        batch_x = iter[0].float().to(self.device)
        batch_y = iter[1].float().to(self.device)

        output = self.model(batch_x)

        output = output[:, -self.args.output_len:, :]
        gt = batch_y

        return output, gt

    def vali(self, loader, criterion):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, iter in tqdm(enumerate(loader), total=len(loader), desc=f"Validating..."):
                    
                    output, gt = self._forward_step(iter)

                    loss = criterion(output, gt)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test(self, loaders, criterion, valinum='full'):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()
        for info, loader in loaders.items():
            info_loss=[]
            with torch.inference_mode():
                with torch.no_grad():
                    for i, iter in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {info}"):
                        
                        output, gt = self._forward_step(iter)

                        loss = criterion(output, gt)
                        total_loss.append(loss.item())
                        info_loss.append(loss.item())
                        ######
                        if valinum == 'full':
                            pass
                        else:
                            if i == valinum:
                                break # only sample 2 batches for faster validation
                        ######
            print(f"Test loss for {info}: {np.average(info_loss)}")
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss