import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_provider.data_factory import Data_Provider


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for time series forecasting.
    
    This class wraps the existing Data_Provider class for compatibility with PyTorch Lightning,
    enabling efficient data loading, multi-GPU training, and automatic data management.
    It handles train/validation/test dataset creation and provides optimized dataloaders
    with support for heterogeneous data and memory-efficient buffering.
    
    Attributes:
        args (object): Configuration arguments containing data parameters
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading
        data_provider (Data_Provider): The underlying data provider instance
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
    """
    def __init__(self, args):
        """
        Initialize the TimeSeriesDataModule with configuration arguments.
        
        Args:
            args (object): Configuration object containing data parameters with attributes:
                - batch_size (int): Number of samples per batch
                - num_workers (int): Number of worker processes for data loading
                - disable_buffer (bool): Whether to disable data buffering for memory efficiency
                - And other data-related configuration parameters required by Data_Provider
        """
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        """
        Initialize datasets based on the current training stage.
        
        This method is called automatically by PyTorch Lightning to set up datasets
        for different stages of the training process. It creates the data provider
        and loads appropriate datasets based on the stage parameter.
        
        Args:
            stage (str, optional): Current stage of training. Can be:
                - 'fit' or None: Sets up training and validation datasets
                - 'test': Sets up test dataset
                - Called on every process when using Distributed Data Parallel (DDP)
        
        Side Effects:
            - Creates self.data_provider if it doesn't exist
            - Loads train_dataset, val_dataset for 'fit' stage
            - Loads test_dataset for 'test' stage or if not already loaded
            - Clears data buffer after setup to free memory
        
        Note:
            The method includes logic to avoid reloading test datasets if they're
            already available, which helps with memory efficiency.
        """
        # Create data provider, reusing the existing implementation
        if not hasattr(self, 'data_provider'):
            self.data_provider = Data_Provider(self.args, buffer=(not self.args.disable_buffer))
        
        if stage == 'fit' or stage is None:
            self.train_dataset = self.data_provider.get_train(return_type='set')
            self.val_dataset = self.data_provider.get_val(return_type='set')
            self.test_dataset = self.data_provider.get_test(return_type='set')
        
        if stage == 'test' or stage is None:
            # check if test dataset is already loaded
            if not hasattr(self, 'test_dataset'):
                self.test_dataset = self.data_provider.get_test(return_type='set')
            else:
                # If test dataset is already loaded, skip loading again
                print("Test dataset already loaded, skipping reloading.")

        
        # Release buffer after setup to free memory
        self.data_provider.data_buffer.clear()

    def train_dataloader(self):
        """
        Return the training dataloader with appropriate configuration.
        
        Creates a DataLoader for the training dataset with shuffling enabled
        and configured to drop the last incomplete batch for consistent
        batch sizes during training.
        
        Returns:
            DataLoader: Training dataloader configured with:
                - shuffle=True for random sampling
                - drop_last=True for consistent batch sizes
                - concat=True for concatenated data handling
        """
        return self.data_provider.get_dataloader(
            self.train_dataset, 
            shuffle=True, 
            drop_last=True, 
            concat=True
        )

    def val_dataloader(self):
        """
        Return the validation dataloader with appropriate configuration.
        
        Creates a DataLoader for the validation dataset without shuffling
        but still dropping incomplete batches for consistent evaluation.
        
        Returns:
            DataLoader: Validation dataloader configured with:
                - shuffle=False for deterministic evaluation
                - drop_last=True for consistent batch sizes
                - concat=True for concatenated data handling
        """
        return self.data_provider.get_dataloader(
            self.val_dataset, 
            shuffle=False, 
            drop_last=True, 
            concat=True
        )

    def test_dataloader(self):
        """
        Return the test dataloader with appropriate configuration.
        
        Creates a DataLoader for the test dataset optimized for evaluation,
        without shuffling or dropping batches to ensure all test samples
        are processed exactly once.
        
        Returns:
            DataLoader or dict: Test dataloader configured with:
                - shuffle=False for deterministic evaluation
                - drop_last=False to include all test samples
                - concat=False for individual dataset handling
                
        Note:
            May return a dictionary of dataloaders if multiple test datasets
            are configured, allowing for evaluation on different test sets.
        """
        # Return a dictionary of dataloaders for each test dataset
        return self.data_provider.get_dataloader(
            self.test_dataset, 
            shuffle=False, 
            drop_last=False, 
            concat=False
        ) 