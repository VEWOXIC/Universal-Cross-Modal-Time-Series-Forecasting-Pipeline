import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from data_provider.data_factory import Data_Provider
from thop import profile


class Exp_Basic(object):
    """
    Base experiment class providing core functionality for time series forecasting experiments.
    
    This abstract base class establishes the fundamental structure and common methods
    for all experiment types in the Universal Cross-Modal Time Series Forecasting Pipeline.
    It handles device management, model initialization, data provider setup, and provides
    templates for training, validation, and testing workflows.
    
    Args:
        args: Configuration object containing experiment parameters including:
            - model configuration (model type, parameters)
            - data configuration (paths, preprocessing settings)
            - training configuration (learning rate, batch size, epochs)
            - device configuration (GPU/CPU, multi-GPU settings)
            - evaluation configuration (metrics, checkpointing)
    
    Attributes:
        args: Experiment configuration object
        device: PyTorch device (CPU or CUDA)
        model: Initialized model moved to appropriate device
        data_provider: Data management object for loading datasets
    
    Example:
        ```python
        # Typically used through subclasses like Experiment
        class CustomExperiment(Exp_Basic):
            def _build_model(self):
                return YourModel(self.args.model_config)
        
        exp = CustomExperiment(args)
        exp.train()
        ```
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer))

    def _build_model(self):
        """
        Abstract method for building the forecasting model.
        
        This method must be implemented by subclasses to specify the model
        architecture and configuration for their specific use case.
        
        Returns:
            torch.nn.Module: Initialized PyTorch model
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Configures and returns the appropriate computing device for training.
        
        Supports both CPU and GPU training with optional multi-GPU setup.
        Sets CUDA_VISIBLE_DEVICES environment variable for GPU device selection.
        
        Returns:
            torch.device: Configured device (CPU or specific CUDA device)
        """
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    
    def _select_optimizer(self):
        """
        Creates and returns the optimizer for model training.
        
        Currently defaults to Adam optimizer with learning rate from configuration.
        Can be extended by subclasses to support different optimizers.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer for model parameters
        """
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        Creates and returns the loss function for model training.
        
        Supports multiple loss functions based on configuration:
        - 'l1': L1Loss (Mean Absolute Error)
        - 'mse': MSELoss (Mean Squared Error)
        
        Returns:
            torch.nn.Module: Configured loss function
        """
        if self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'mse':
            return nn.MSELoss()
        
    
    def _get_profile(self, model):
        """
        Analyzes model computational complexity and parameter count.
        
        Uses the thop library to calculate FLOPs (Floating Point Operations)
        and parameter count for the model with sample input data.
        
        Args:
            model (torch.nn.Module): Model to profile
        
        Returns:
            tuple: (macs, params) where macs is FLOPs count and params is parameter count
        """
        _input = torch.randn(self.args.batch_size, self.args.seq_len, 1).to(self.device)
        macs, params = profile(model, inputs=(_input).to(self.device))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def _get_data(self):
        """
        Abstract method for data retrieval.
        
        This method should be implemented by subclasses to define how
        training, validation, and test data are obtained and formatted.
        """
        pass

    def vali(self):
        """
        Abstract method for model validation.
        
        This method should be implemented by subclasses to define the
        validation procedure and metrics calculation.
        """
        pass

    def train(self):
        """
        Abstract method for model training.
        
        This method should be implemented by subclasses to define the
        complete training loop including optimization and checkpointing.
        """
        pass

    def test(self):
        """
        Abstract method for model testing.
        
        This method should be implemented by subclasses to define the
        testing procedure and final evaluation metrics.
        """
        pass
