# PyTorch Lightning for Time Series Forecasting

This extension provides a PyTorch Lightning implementation of the time series forecasting pipeline, making it easier to leverage multi-GPU training, mixed precision, and other advanced training features.

## Benefits of the PyTorch Lightning Implementation

- **Multi-GPU Training**: Easily distribute training across multiple GPUs with minimal code changes.
- **Mixed Precision Training**: Reduce memory usage and speed up training with 16-bit precision.
- **Better Code Organization**: Clear separation between model logic, data processing, and training loop.
- **Built-in Features**: Access to features like gradient clipping, early stopping, and model checkpointing.
- **Experiment Tracking**: Native TensorBoard integration for experiment monitoring.

## How to Use

### Running a Training Job

```bash
python run_lightning.py --model FITS --data solar --input_len 96 --output_len 96 --use_multi_gpu --devices 0,1,2,3
```

### Key Parameters

- **Model Configuration**: `--model`, `--model_config`
- **Data Configuration**: `--data`, `--data_config`, `--input_len`, `--output_len`
- **Training Parameters**: `--train_epochs`, `--batch_size`, `--learning_rate`, `--patience`
- **GPU Options**: `--use_gpu`, `--gpu`, `--use_multi_gpu`, `--devices`
- **Lightning-Specific**: `--precision`, `--gradient_clip_val`

### Sample Multi-GPU Command

```bash
python run_lightning.py \
    --model DLinear \
    --data ETTh1 \
    --input_len 96 \
    --output_len 96 \
    --use_multi_gpu \
    --devices 0,1,2,3 \
    --precision 16 \
    --batch_size 256
```

## Directory Structure

```
├── data_provider/
│   ├── data_factory.py        # Original data provider
│   ├── data_loader.py         # Original dataset loader
│   └── lightning_data_module.py # Lightning data module wrapper
├── exp/
│   ├── exp_universal.py       # Original training pipeline
│   └── exp_lightning.py       # Lightning model and training logic
├── run.py                     # Original run script
└── run_lightning.py           # PyTorch Lightning run script
```

## Implementation Details

The PyTorch Lightning implementation wraps the existing models and datasets with Lightning components:

1. **TimeSeriesDataModule**: Wraps the existing `Data_Provider` to create PyTorch Lightning-compatible data loaders
2. **TimeSeriesLightningModel**: Wraps the model implementation and training logic in a Lightning module
3. **train_lightning_model**: Orchestrates the training process with Lightning Trainer

The implementation is designed to be compatible with the existing codebase, so you can use the same models and configurations as before.

## Advanced Usage

### Mixed Precision Training

```bash
python run_lightning.py --model FITS --data solar --precision 16
```

### Gradient Clipping

```bash
python run_lightning.py --model FITS --data solar --gradient_clip_val 0.5
```

### Changing Early Stopping Criteria

Modify the `EarlyStopping` callback in `exp_lightning.py` to change the early stopping criteria.

### Custom Learning Rate Schedules

The implementation already supports the existing learning rate adjustment strategies. For custom schedules, modify the `configure_optimizers` method in the `TimeSeriesLightningModel` class. 