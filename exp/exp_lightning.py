import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from models import model_init
from utils.tools import general_move_to_device, adjust_learning_rate
import json
import time
from tqdm import tqdm
import warnings
import inspect

warnings.filterwarnings('ignore')


class TimeSeriesLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for time series forecasting.
    Wraps the existing model implementations and training logic.
    """
    def __init__(self, args):
        super(TimeSeriesLightningModel, self).__init__()
        self.args = args
        self.save_hyperparameters(ignore=['args'])
        
        # Build model
        self.model = model_init(self.args.model, self.args.model_config, self.args)
        
        # Loss function
        self.criterion = self._select_criterion()
        
        # Track metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = {}  # Dictionary to track per-dataset metrics
        
    def _select_criterion(self):
        """Select the loss function."""
        if self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'mse':
            return nn.MSELoss()
        else:
            return nn.MSELoss()  # Default
    
    def forward(self, batch):
        """Forward pass."""
        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = batch
        
        if hasattr(self.model, 'move_to_device'):
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(
                batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, 
                hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device
            )
        else:
            # Only move batch_x, batch_y to device for TSF models
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
        
        # Inspect model's forward method signature to determine what inputs to provide
        model_params = list(inspect.signature(self.model.forward).parameters.keys())
        
        # Build input dictionary based on available parameters
        forward_kwargs = {}
        if 'x' in model_params:
            forward_kwargs['x'] = batch_x
        if 'historical_events' in model_params and 'historical_events' not in forward_kwargs:
            forward_kwargs['historical_events'] = batch_x_hetero
        if 'news' in model_params and 'news' not in forward_kwargs:
            forward_kwargs['news'] = batch_y_hetero
        if 'dataset_description' in model_params and 'dataset_description' not in forward_kwargs:
            forward_kwargs['dataset_description'] = hetero_general
        if 'channel_description' in model_params and 'channel_description' not in forward_kwargs:
            forward_kwargs['channel_description'] = hetero_channel
        
        # Call model with appropriate arguments
        output = self.model(**forward_kwargs)
        
        output = output[:, -self.args.output_len:, :]
        return output, batch_y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """After validation completes, run test on all subsets."""
        # Only run test during training, not during sanity check
        if self.trainer.sanity_checking:
            return
        
        # Manually run test on all subsets
        self._run_epoch_test()
        
    def _run_epoch_test(self):
        """Run test on all subsets and print results."""
        # Skip if we don't have access to datamodule or if test dataset isn't set up yet
        if not hasattr(self.trainer, 'datamodule') or not hasattr(self.trainer.datamodule, 'test_dataset'):
            return
        
        print("\n\n------- Testing on Epoch {} -------".format(self.current_epoch + 1))
        
        # Save current state
        self.model.eval()
        test_loaders = self.trainer.datamodule.test_dataloader()
        
        # Track losses
        subset_losses = {}
        all_losses = []
        
        with torch.no_grad():
            for subset_id, loader in test_loaders.items():
                subset_batch_losses = []
                
                # Process each batch
                for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {info}"):
                    output, gt = self.forward(batch)
                    loss = self.criterion(output, gt)
                    subset_batch_losses.append(loss.item())
                
                # Calculate average for this subset
                if subset_batch_losses:
                    avg_loss = np.mean(subset_batch_losses)
                    subset_losses[subset_id] = avg_loss
                    all_losses.extend(subset_batch_losses)
                    print(f"Test loss for {subset_id}: {avg_loss:.7f}")
        
        # Calculate overall average
        if all_losses:
            overall_avg = np.mean(all_losses)
            print(f"Overall test loss: {overall_avg:.7f}")
            
            # Log to tensorboard
            self.log('epoch_test_loss', overall_avg, on_epoch=True)
            
            # Log each subset
            for subset_id, loss in subset_losses.items():
                self.log(f'epoch_test_loss_{subset_id}', loss, on_epoch=True)
        
        print("---------------------------------------\n")
        
        # Restore model state
        self.model.train()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # Get dataset identifier for the current dataloader
        dataset_ids = list(self.trainer.datamodule.test_dataset.keys())
        if dataloader_idx < len(dataset_ids):
            dataset_id = dataset_ids[dataloader_idx]
        else:
            dataset_id = f"unknown_{dataloader_idx}"
        
        # Initialize list for this dataset if it doesn't exist
        if dataset_id not in self.test_step_outputs:
            self.test_step_outputs[dataset_id] = []
        
        # Store the loss for this batch in the corresponding dataset
        self.test_step_outputs[dataset_id].append(loss.item())
        
        # Log individual test losses
        self.log(f'test_loss_{dataset_id}', loss, on_epoch=True)
        
        return {"loss": loss, "dataset_id": dataset_id}
    
    def on_test_epoch_end(self):
        """
        Compute and log the average test loss across all datasets.
        This mimics the behavior of the PyTorch implementation where 
        each dataset's average loss is reported separately.
        """
        # Calculate average loss for each dataset
        all_losses = []
        
        for dataset_id, losses in self.test_step_outputs.items():
            avg_loss = torch.tensor(losses).mean().item()
            all_losses.extend(losses)  # Collect all losses for overall average
            
            # Log average loss for this dataset
            self.log(f'test_avg_loss_{dataset_id}', avg_loss)
            print(f"Test loss for {dataset_id}: {avg_loss:.7f}")
        
        # Calculate overall average loss
        if all_losses:
            overall_avg_loss = torch.tensor(all_losses).mean().item()
            self.log('test_avg_loss', overall_avg_loss)
            print(f"Overall test loss: {overall_avg_loss:.7f}")
        
        # Clear the outputs
        self.test_step_outputs = {}
    
    def configure_optimizers(self):
        """Configure optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        
        # Custom learning rate adjustment
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda epoch: adjust_learning_rate(None, epoch, self.args, return_rate=True)
            ),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [lr_scheduler]


class TestAfterEpochCallback(pl.Callback):
    """
    Callback to run test after each epoch and display per-subset results.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        """Run test after validation epoch ends."""
        pass  # The testing is now handled directly in the model's on_validation_epoch_end


def train_lightning_model(args, setting):
    """
    Train the Lightning model and save it.
    
    Args:
        args: Arguments for the experiment
        setting: String identifier for the experiment
    
    Returns:
        Trained model
    """
    from data_provider.lightning_data_module import TimeSeriesDataModule
    
    # Initialize data module
    data_module = TimeSeriesDataModule(args)
    
    # Initialize model
    model = TimeSeriesLightningModel(args)
    
    # Create checkpoint directory
    checkpoint_path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Save args
    with open(os.path.join(checkpoint_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    
    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    # Add test-after-epoch callback
    test_callback = TestAfterEpochCallback()
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.checkpoints, 'tb_logs'),
        name=setting
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator='gpu' if args.use_gpu else 'cpu',
        devices=args.device_ids if args.use_multi_gpu else [args.gpu] if args.use_gpu else None,
        strategy='ddp' if args.use_multi_gpu else None,
        callbacks=[early_stopping, checkpoint_callback, test_callback],  # Added test callback
        logger=logger,
        deterministic=True,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
    )
    
    # Train model
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.fit(model, data_module)
    
    # Final test
    print(f'>>>>>>>final testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.test(model, data_module)
    
    # Load best model and return
    best_model_path = checkpoint_callback.best_model_path
    best_model = TimeSeriesLightningModel.load_from_checkpoint(best_model_path)
    
    return best_model.model  # Return the wrapped model for compatibility 