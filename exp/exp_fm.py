from exp.exp_basic import Exp_Basic
from models import model_init

import numpy as np
import torch
import torch.nn as nn

import os
import time
import warnings

import json
from tqdm import tqdm
from data_provider.data_factory import Data_Provider

from utils.tools import general_move_to_device

warnings.filterwarnings('ignore')


class Experiment(Exp_Basic):
    
    def __init__(self, args):
        super(Experiment, self).__init__(args)
        
    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args, is_FM=True)
        return model

    def _get_data(self, flag, return_type='loader'):
        """
        Get the data for training, validation, or testing.
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type=return_type)
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type=return_type)
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type=return_type)

        return data_loader

    def _forward_step(self, iter):
        """
        Forward step for the model.
        """
        # iteration: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter

        if hasattr(self.model, 'move_to_device'):
            # move only the ones needed to device according to model's definition to save VRAM
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        
        num_channels = batch_x.size(-1)
        outputs = []
        for c in range(num_channels):
            # batch_x: [batch_size, seq_len, num_channels]
            channel_x = batch_x[:, :, c:c+1].squeeze(-1)  # channel_x: [batch_size, seq_len]
            channel_output = self.model(x=channel_x).unsqueeze(-1)  # channel_output: [batch_size, output_len, 1]
            outputs.append(channel_output)
        
        final_output = torch.cat(outputs, dim=-1)  # final_output: [batch_size, output_len, num_channels]
        gt = batch_y  # batch_y: [batch_size, output_len, num_channels]

        return final_output, gt

    def test(self, savepath):
        """
        Validate the model on the validation dataset.
        """
        path = os.path.join(self.args.checkpoints, savepath)
        if not os.path.exists(path):
            os.makedirs(path)
        
        all_metrics_filename = "all_test_metrics.json"
        final_filename = "final_test_result.json"
        
        criterion = self._select_criterion()
        loaders = self._get_data(flag='test')

        overall_running_loss = 0.0
        overall_total_samples = 0

        self.model.eval()

        for info, loader in loaders.items():
            info_running_loss = 0.0
            info_total_samples = 0
            
            with torch.inference_mode():
                for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {info}"):
                    
                    output, gt = self._forward_step(iter_data)
                    
                    current_batch_size = gt.size(0)
                    loss = criterion(output, gt)

                    info_running_loss += loss.item() * current_batch_size
                    info_total_samples += current_batch_size
                    
                    overall_running_loss += loss.item() * current_batch_size
                    overall_total_samples += current_batch_size
            
            if info_total_samples > 0:
                info_epoch_loss = info_running_loss / info_total_samples
                print(f"Test loss for {info}: {info_epoch_loss:.7f}")
            else:
                print(f"Test loss for {info}: N/A (no samples processed)")
                info_epoch_loss = None
            
            # Save metrics
            try:
                with open(os.path.join(path, all_metrics_filename), 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}

            existing_data.update({info: info_epoch_loss})
            with open(os.path.join(path, all_metrics_filename), 'w') as f:
                json.dump(existing_data, f, indent=4)

        total_epoch_loss = overall_running_loss / overall_total_samples if overall_total_samples > 0 else 0.0
        print(f"Overall test loss: {total_epoch_loss:.7f}")

        # Save results
        final_result = {"final_res": total_epoch_loss}
        
        with open(os.path.join(path, final_filename), 'w') as f:
            json.dump(final_result, f, indent=4)
        print(f"Saved final result to {final_filename}")
