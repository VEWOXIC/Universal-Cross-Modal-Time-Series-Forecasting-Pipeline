import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_provider.data_factory import Data_Provider


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for time series forecasting.
    Wraps the existing Data_Provider class for compatibility with PyTorch Lightning.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        """
        Initialize datasets based on stage.
        Called on every process when using DDP.
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
        """Return the training dataloader."""
        return self.data_provider.get_dataloader(
            self.train_dataset, 
            shuffle=True, 
            drop_last=True, 
            concat=True
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.data_provider.get_dataloader(
            self.val_dataset, 
            shuffle=False, 
            drop_last=True, 
            concat=True
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        # Return a dictionary of dataloaders for each test dataset
        return self.data_provider.get_dataloader(
            self.test_dataset, 
            shuffle=False, 
            drop_last=False, 
            concat=False
        ) 