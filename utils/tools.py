import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, return_rate=False):
    """
    Adjust the learning rate during training based on the specified strategy.
    
    This function implements various learning rate adjustment strategies to improve
    training convergence and model performance. It supports multiple predefined
    strategies and can either modify the optimizer directly or return a rate
    multiplier for use with PyTorch Lightning.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        epoch (int): Current training epoch number.
        args (object): Arguments object containing learning rate configuration with attributes:
            - learning_rate (float): Initial learning rate
            - lradj (str): Learning rate adjustment strategy ('type1', 'type2', 'type3', 'type4', 'constant')
        return_rate (bool, optional): If True, returns the rate multiplier instead of
                                    modifying the optimizer directly. Useful for PyTorch Lightning.
                                    Defaults to False.
    
    Returns:
        float or None: If return_rate is True, returns the learning rate multiplier.
                      Otherwise, returns None and modifies the optimizer directly.
    
    Raises:
        ValueError: If the specified learning rate adjustment strategy is not supported.
    
    Strategies:
        - 'type1': Halve the learning rate every epoch
        - 'type2': Predefined schedule with specific rates at certain epochs
        - 'type3': Constant for first 3 epochs, then decay by 0.9 each epoch
        - 'type4': Decay by 0.9 every epoch from the start
        - 'constant': Keep learning rate constant throughout training
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.50 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    else:
        raise ValueError('Learning rate adjustment strategy not supported')
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        # If a multiplier rate is needed for PyTorch Lightning's LambdaLR
        if return_rate:
            return lr / args.learning_rate
        # Otherwise, adjust optimizer directly
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        # If we're just returning the rate for PyTorch Lightning
        if return_rate:
            return None if epoch not in lr_adjust.keys() else lr_adjust[epoch] / args.learning_rate


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during model training.
    
    This class monitors validation loss and stops training when the loss stops
    improving for a specified number of epochs (patience). It also handles
    model checkpointing by saving the best model encountered during training.
    
    Attributes:
        patience (int): Number of epochs to wait without improvement before stopping
        verbose (bool): Whether to print messages about validation loss improvements
        counter (int): Current count of epochs without improvement
        best_score (float): Best validation score encountered so far
        early_stop (bool): Flag indicating whether to stop training
        val_loss_min (float): Minimum validation loss encountered
        delta (float): Minimum change required to qualify as an improvement
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the EarlyStopping monitor.
        
        Args:
            patience (int, optional): Number of epochs to wait for improvement before
                                    stopping training. Defaults to 7.
            verbose (bool, optional): If True, prints messages when validation loss
                                    improves and model is saved. Defaults to False.
            delta (float, optional): Minimum change in validation loss to qualify as
                                   an improvement. Must be positive. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Check if training should stop based on validation loss and save model if improved.
        
        This method is called after each epoch to evaluate whether training should
        continue. It compares the current validation loss with the best seen so far
        and updates counters accordingly.
        
        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): The model to be saved if improvement is detected.
            path (str): Directory path where the model checkpoint should be saved.
        
        Side Effects:
            - Updates self.early_stop flag if patience is exceeded
            - Saves model checkpoint if validation loss improves
            - Updates internal counters and best score tracking
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save the model checkpoint when validation loss improves.
        
        This method saves the model's state dictionary to a checkpoint file
        when a new best validation loss is achieved.
        
        Args:
            val_loss (float): Current validation loss that represents an improvement.
            model (torch.nn.Module): The model whose state dict will be saved.
            path (str): Directory path where the checkpoint file will be saved.
                       The file will be saved as 'checkpoint.pth' in this directory.
        
        Side Effects:
            - Saves model state dict to '{path}/checkpoint.pth'
            - Updates self.val_loss_min with the new minimum validation loss
            - Prints improvement message if verbose mode is enabled
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



class dotdict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    
    This class extends the built-in dict class to provide convenient dot notation
    access to dictionary items, making configuration objects more intuitive to use.
    It supports getting, setting, and deleting attributes as if they were regular
    object attributes while maintaining full dictionary functionality.
    
    Example:
        >>> config = dotdict({'learning_rate': 0.001, 'epochs': 100})
        >>> config.learning_rate  # Returns 0.001
        >>> config.batch_size = 32  # Sets config['batch_size'] = 32
        >>> del config.epochs  # Deletes config['epochs']
    """
    def __getattr__(self, name):
        """
        Get dictionary value using attribute notation.
        
        Args:
            name (str): The key name to retrieve from the dictionary.
        
        Returns:
            object: The value associated with the key, or None if key doesn't exist.
        """
        if name in self.keys():

            return self[name]
        else:
            return None
    def __setattr__(self, name, value):
        """
        Set dictionary value using attribute notation.
        
        Args:
            name (str): The key name to set in the dictionary.
            value (object): The value to associate with the key.
        """
        self[name] = value
    def __delattr__(self, name):
        """
        Delete dictionary key using attribute notation.
        
        Args:
            name (str): The key name to delete from the dictionary.
        
        Raises:
            KeyError: If the key doesn't exist in the dictionary.
        """
        del self[name]


class StandardScaler():
    """
    A standard scaler for normalizing data using pre-computed mean and standard deviation.
    
    This class implements z-score normalization (standardization) using provided
    mean and standard deviation values. Unlike sklearn's StandardScaler, this
    implementation uses pre-computed statistics rather than computing them from data.
    
    Attributes:
        mean (float or array-like): The mean value(s) used for normalization
        std (float or array-like): The standard deviation value(s) used for normalization
    """
    def __init__(self, mean, std):
        """
        Initialize the StandardScaler with pre-computed statistics.
        
        Args:
            mean (float or array-like): Mean value(s) to subtract during transformation.
                                      Should match the dimensionality of the data to be transformed.
            std (float or array-like): Standard deviation value(s) to divide by during transformation.
                                     Should match the dimensionality of the data to be transformed.
                                     Must be non-zero to avoid division by zero.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Apply standardization to the input data.
        
        Transforms the data by subtracting the mean and dividing by the standard deviation:
        transformed_data = (data - mean) / std
        
        Args:
            data (array-like): Input data to be standardized. Should be compatible
                             with the mean and std provided during initialization.
        
        Returns:
            array-like: Standardized data with zero mean and unit variance (approximately).
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverse the standardization transformation.
        
        Converts standardized data back to the original scale by multiplying by
        the standard deviation and adding the mean:
        original_data = (data * std) + mean
        
        Args:
            data (array-like): Standardized data to be converted back to original scale.
        
        Returns:
            array-like: Data in the original scale before standardization.
        """
        return (data * self.std) + self.mean

import psutil, os

def log_memory_usage():
    """
    Log the current memory usage of the running process.
    
    This utility function prints the memory consumption of the current Python process
    in megabytes. It's useful for monitoring memory usage during training or debugging
    memory-related issues.
    
    Side Effects:
        Prints a message showing the process ID and memory usage in MB to the console.
    
    Example:
        >>> log_memory_usage()
        Memory usage 12345: 1024.50 MB
    """
    process = psutil.Process(os.getpid())
    print(f"Memory usage {os.getpid()}: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
    """
    Move time series data tensors to the specified computing device.
    
    This utility function handles the transfer of multiple data tensors from CPU to GPU
    or between different devices. It's specifically designed for time series forecasting
    workflows that involve both homogeneous time series data and heterogeneous auxiliary data.
    
    Args:
        batch_x (torch.Tensor): Input time series data tensor.
        batch_y (torch.Tensor): Target time series data tensor.
        timestamp_x (object): Timestamps for input data (not moved to device).
        timestamp_y (object): Timestamps for target data (not moved to device).
        batch_x_hetero (object): Heterogeneous input data (not moved to device).
        batch_y_hetero (object): Heterogeneous target data (not moved to device).
        hetero_x_time (object): Heterogeneous input time data (not moved to device).
        hetero_y_time (object): Heterogeneous target time data (not moved to device).
        hetero_general (object): General heterogeneous data (not moved to device).
        hetero_channel (object): Channel-specific heterogeneous data (not moved to device).
        device (str or torch.device): Target device for tensor placement (e.g., 'cuda:0', 'cpu').
    
    Returns:
        tuple: A tuple containing all input arguments with batch_x and batch_y moved to
               the specified device as float tensors, while other arguments remain unchanged.
    
    Note:
        Only batch_x and batch_y tensors are moved to the device and converted to float.
        All other arguments (timestamps, heterogeneous data) are returned as-is.
    """
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    return batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel