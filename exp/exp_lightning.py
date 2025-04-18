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
        
        # Configure automatic optimization if needed
        self.automatic_optimization = True
        
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
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    def on_validation_epoch_end(self):
        """After validation completes, run test on all subsets."""
        # Only run test during training, not during sanity check
        if self.trainer.sanity_checking:
            return
        
        # Manually run test on all subsets
        if self.args.test_after_epoch:
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
                for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {subset_id}"):
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
            
        
        print("---------------------------------------\n")
        
        # Restore model state
        self.model.train()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)

        self.log(f'test_loss_dataloader_{dataloader_idx}', loss, on_epoch=True, add_dataloader_idx=False, sync_dist=True)

        return {"loss": loss, "dataloader_idx": dataloader_idx}

    # def on_test_epoch_end(self):
    #     metrics = self.trainer.callback_metrics
        
    #     # Find all test loss metrics for different dataloaders
    #     test_losses = {k: v.item() for k, v in metrics.items() if k.startswith('test_loss_dataloader_')}
        
    #     # Log overall average loss
    #     if test_losses:
    #         overall_avg_loss = sum(test_losses.values()) / len(test_losses)
    #         self.log('test_avg_loss', overall_avg_loss)
    #         print(f"Overall test loss: {overall_avg_loss:.7f}")


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
    
    # Advanced trainer configurations for better performance
    trainer_kwargs = {
        'max_epochs': args.train_epochs,
        'accelerator': 'gpu' if args.use_gpu else 'cpu',
        'devices': args.device_ids if args.use_multi_gpu else [args.gpu] if args.use_gpu else None,
        'strategy': 'ddp' if args.use_multi_gpu else None,
        'callbacks': [early_stopping, checkpoint_callback],
        'logger': logger,
        'deterministic': True,
        'precision': getattr(args, 'precision', 32),
        'gradient_clip_val': args.gradient_clip_val if hasattr(args, 'gradient_clip_val') and args.gradient_clip_val > 0 else None,
        # Added for better performance
        'num_sanity_val_steps': 0,  # Skip sanity check for faster startup
        'enable_checkpointing': True,
        'enable_model_summary': True,
        'enable_progress_bar': True,
        'log_every_n_steps': 50,
    }
    
    # Add profiler if requested
    if hasattr(args, 'profiler') and args.profiler:
        trainer_kwargs['profiler'] = 'simple'
    
    # Configure trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.fit(model, data_module)
    
    # Final test - run it once with all dataloaders together
    print(f'>>>>>>>final testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    test_loaders = data_module.test_dataloader()
    dataloaders_list = [loader for _, loader in test_loaders.items()]
    dataset_ids = list(test_loaders.keys())
    
    results = trainer.test(model, dataloaders=dataloaders_list)
    
    # Print results for each dataset
    for i, result in enumerate(results):
        dataset_id = dataset_ids[i] if i < len(dataset_ids) else f"unknown_{i}"
        print(f"Test results for {dataset_id}:")
        print(f"- Loss: {result[f'test_loss_dataloader_{i}']:.7f}")
    
    # Load best model and return
    best_model_path = checkpoint_callback.best_model_path

    print(f"Best model path: {best_model_path}")
    
    return best_model_path  # Return the wrapped model for compatibility 