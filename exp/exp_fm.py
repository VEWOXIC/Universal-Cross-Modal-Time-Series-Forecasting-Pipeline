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


class Experiment(Exp_Basic):
    """
    Experiment orchestrator for Foundation Model (FM) based time series forecasting.
    
    This class specializes the base experiment framework for pre-trained foundation models
    that have been designed for time series forecasting tasks. Foundation models leverage
    large-scale pre-training on diverse time series data to provide strong generalization
    capabilities across different domains and datasets.
    
    Key Features:
        - Foundation model initialization with pre-trained weights
        - Efficient fine-tuning and adaptation workflows
        - Memory-optimized device management for large models
        - Support for cross-modal and heterogeneous data inputs
        - Specialized inference procedures for foundation models
    
    Args:
        args: Configuration object containing FM-specific parameters including:
            - model_config: Foundation model architecture and parameter settings
            - fine_tuning: Adaptation strategy and learning rate schedules
            - inference: Batch processing and memory optimization settings
            - data: Cross-modal data configuration and preprocessing
    
    Example:
        ```python
        exp = Experiment(args)
        model = exp.train(setting='fm_adaptation_v1') 
        results = exp.test(setting='fm_adaptation_v1')
        ```
    """
    
    def __init__(self, args):
        super(Experiment, self).__init__(args)
        
    def _build_model(self):
        """
        Constructs and returns a Foundation Model for time series forecasting.
        
        Initializes pre-trained foundation models with specialized configurations
        for time series forecasting tasks. The model is built with FM-specific
        optimizations and adaptation capabilities.
        
        Returns:
            torch.nn.Module: Initialized foundation model configured for time series forecasting
        """
        model = model_init(self.args.model, self.args.model_config, self.args, is_FM=True)
        return model

    def _get_data(self, flag, return_type='loader'):
        """
        Retrieves data in the specified format for foundation model experiments.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
            return_type (str): Format of returned data ('loader', 'set', 'both')
        
        Returns:
            DataLoader/Dataset/tuple: Data in requested format for foundation model training/inference
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
        Executes a single forward pass for foundation model inference.
        
        Handles device placement and data formatting specific to foundation models,
        which may have different memory requirements and optimization strategies
        compared to standard models.
        
        Args:
            iter: Data batch containing time series and cross-modal information
        
        Returns:
            tuple: (predictions, ground_truth) for foundation model evaluation
        """
        # iteration: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter

        if hasattr(self.model, 'move_to_device'):
            # move only the ones needed to device according to model's definition to save VRAM
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        
        if self.args.individual:
            num_channels = batch_x.size(-1)
            outputs = []
            for c in range(num_channels):
                # batch_x: [batch_size, seq_len, num_channels]
                channel_x = batch_x[:, :, c:c+1].squeeze(-1)  # channel_x: [batch_size, seq_len]

                if self.args.task == 'TSF':
                    channel_output = self.model.forward(x=channel_x)  # channel_output: [batch_size, output_len, 1]
                elif self.args.task == 'TGTSF':
                    channel_output = self.model.forward(x=channel_x, batch_y_hetero=batch_y_hetero, hetero_general=hetero_general, hetero_channel=hetero_channel)
                else:
                    raise ValueError(f"Unsupported task type: {self.args.task}")
                
                if channel_output is None:
                    print(f"[ Warning ]: Model returned None for channel {c}.")
                    return None, None
                elif torch.isnan(channel_output).any():
                    print(f"[ Warning ]: NaN detected in channel {c} output")
                    return None, None
                else:
                    channel_output = channel_output.unsqueeze(-1)
                    print(f"Channel {c} output shape: {channel_output}")
                
                outputs.append(channel_output)
            
            final_output = torch.cat(outputs, dim=-1)  # final_output: [batch_size, output_len, num_channels]
        
        else:
            if self.args.task == 'TSF':
                final_output = self.model.forward(x = batch_x)
            elif self.args.task == 'TGTSF':
                final_output = self.model.forward(x=channel_x, batch_y_hetero=batch_y_hetero, hetero_general=hetero_general, hetero_channel=hetero_channel)
            else:
                raise ValueError(f"Unsupported task type: {self.args.task}")
            
            if final_output is None:
                print(f"[ Warning ]: Model returned None.")
                return None, None
            elif torch.isnan(channel_output).any():
                print(f"[ Warning ]: NaN detected in model output")
                return None, None

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
        error_filename = "overall_error.json"
        
        criterion = self._select_criterion()
        loaders = self._get_data(flag='test')

        overall_running_loss = 0.0
        overall_total_samples = 0
        overall_error = 0

        if self.args.filtered_samples is not None:
            filtered_samples = json.load(open(self.args.filtered_samples))
            print(f"[Info] Using filtered samples from: {self.args.filtered_samples}")

        self.model.eval()

        for info, loader in loaders.items():
            info_running_loss = 0.0
            info_total_samples = 0
            info_error = 0
            
            if self.args.filtered_samples is not None:
                filter_index = filtered_samples[info]

            with torch.inference_mode():
                for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {info}"):
                    if self.args.filtered_samples is not None and i in filter_index:

                        print(f"[ Info ]: Testing on sample {i}, total: {len(filter_index)}")

                        output, gt = self._forward_step(iter_data)

                        if output is None and gt is None:
                            print(f"[ Warning ]: Model returned None for sample {i}. Skipping this sample.")
                            info_error += 1
                            overall_error += 1
                            continue
                        
                        current_batch_size = gt.size(0)
                        loss = criterion(output, gt)

                        info_running_loss += loss.item() * current_batch_size
                        info_total_samples += current_batch_size
                        
                        overall_running_loss += loss.item() * current_batch_size
                        overall_total_samples += current_batch_size
                    
                    elif self.args.filtered_samples is None:

                        print(f"[ Info ]: Testing on all samples")

                        output, gt = self._forward_step(iter_data)

                        if output is None and gt is None:
                            print(f"[ Warning ]: Model returned None for sample {i}. Skipping this sample.")
                            info_error += 1
                            overall_error += 1
                            continue
                        
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

            print(f"Total Errors: {info_error}")
            
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

        # Save overall errors
        overall_error_result = {"overall_error": overall_error}
        with open(os.path.join(path, error_filename), 'w') as f:
            json.dump(overall_error_result, f, indent=4)
        print(f"Saved overall error info to {error_filename}")
