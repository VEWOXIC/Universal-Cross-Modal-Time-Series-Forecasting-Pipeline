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
        self.test_step_outputs = []
        
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
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # If we're testing with multiple dataloaders (one per test dataset)
        # Keep track of which one we're using
        self.log(f'test_loss_{dataloader_idx}', loss, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log the average test loss across all dataloaders."""
        # This will be called after all test batches have been processed
        # We can calculate and log average metrics here if needed
        pass
    
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
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        deterministic=True,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
    )
    
    # Train model
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.fit(model, data_module)
    
    # Test model
    print(f'>>>>>>>testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.test(model, data_module)
    
    # Load best model and return
    best_model_path = checkpoint_callback.best_model_path
    best_model = TimeSeriesLightningModel.load_from_checkpoint(best_model_path)
    
    return best_model.model  # Return the wrapped model for compatibility 