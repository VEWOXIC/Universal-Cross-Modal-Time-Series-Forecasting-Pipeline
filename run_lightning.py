import argparse
import os
import torch
from exp.exp_lightning import train_lightning_model
import random
import numpy as np
import time
import yaml
from utils.tools import dotdict
from utils.task import ahead_task_parser

parser = argparse.ArgumentParser(description='Time Series Forecasting with PyTorch Lightning - A flexible framework supporting various forecasting models and datasets.')

# model config
parser.add_argument('--model', type=str, default='FITS', help='Model name (DLinear, FITS, PatchTST, TGTSF, iTransformer, etc.) - Must correspond to a model file in models/ directory')
parser.add_argument('--model_config', type=str, default='model_configs/FITS.yaml', help='Path to model configuration YAML file, containing model-specific parameters')
parser.add_argument('--last_ckpt', type=str, default=None, help='Path to the last checkpoint file (if resuming training or testing from a specific checkpoint)')

# data loader
parser.add_argument('--data', type=str, default='solar', help='Dataset name for reference (actual data location is specified in data_config)')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory to save model checkpoints, TensorBoard logs, and training artifacts')
parser.add_argument('--data_config', type=str, default='./data_configs/data_profile.yaml', help='Path to data configuration YAML file specifying dataset location, format, and preprocessing')
parser.add_argument('--scale', type=bool, default=True, help='Whether to standardize the data (zero mean, unit variance)')
parser.add_argument('--disable_buffer', default=False, action='store_true', help='Disable data buffer to reduce memory usage (may slow down training)')
parser.add_argument('--preload_hetero', default=False, action='store_true', help='Preload heterogeneous data for faster access (increases RAM usage but reduces disk I/O)')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch per worker in dataloader (higher values use more memory)')
parser.add_argument('--noise', type=float, default=0.0, help='optimizer learning rate')
parser.add_argument('--downsample', type=int, default=1, help='number of augmented data')

# forecasting task
parser.add_argument('--ahead', type=str, default=None, help='Shorthand for forecast horizon: "day", "week", or "month" (automatically sets input_len and output_len based on sampling_rate)')
parser.add_argument('--output_len', type=int, default=1000, help='Output/prediction sequence length (number of time steps to forecast)')
parser.add_argument('--input_len', type=int, default=1000, help='Input sequence length (number of historical time steps used for prediction)')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='Number of subprocesses for data loading (0 means data is loaded in the main process)')
parser.add_argument('--train_epochs', type=int, default=50, help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=96, help='Batch size for training (per GPU when using multi-GPU)')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience: training stops if validation loss does not improve for this many epochs')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate for optimizer')
parser.add_argument('--loss', type=str, default='mse', help='Loss function: "mse" (Mean Squared Error) or "l1" (Mean Absolute Error)')
parser.add_argument('--lradj', type=str, default='type3', help='Learning rate adjustment strategy: "type1" (halving), "type2" (step schedule), "type3" (cosine decay), "type4" (linear decay), or "constant"')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU for training (if available)')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use when not using multi-GPU')
parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs for distributed training with DDP', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='Comma-separated list of GPU device IDs to use for multi-GPU training')

# PyTorch Lightning specific
parser.add_argument('--precision', type=str, default='32', help='Numerical precision for training: "32" (float32), "16" (float16/half), or "bf16" (bfloat16)')
parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value to prevent exploding gradients (0 to disable)')
parser.add_argument('--test_after_epoch', action='store_true', help='Test after each epoch')

# Testing
parser.add_argument('--test', action='store_true', help='Test the model after training (if set, training will not be performed). need specify the last checkpoint')

args = parser.parse_args()

# make training faster
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


# preload the yamls
with open(args.model_config, 'r') as f:
    model_config = yaml.safe_load(f)
model_config = dotdict(model_config)
args.model_config = model_config

with open(args.data_config, 'r') as f:
    data_configs = yaml.safe_load(f)
data_configs = dotdict(data_configs)
args.data_config = data_configs

# get current time
current_time = 'ligntning_' + time.strftime("%Y%m%d_%H", time.localtime())

# setting record of experiment
if args.ahead is not None:
    assert args.ahead in ['day', 'week', 'month'], 'ahead task not supported, or add your own parser'

    try:
        args.output_len, args.input_len = ahead_task_parser(args.ahead, data_configs.sampling_rate)
        setting = f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead_pl'  # Added _pl suffix to identify Lightning runs
    except:
        setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}_pl'
        raise ValueError('sampling rate not found in data config, fall back to default, input output length')
else:
    setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}_pl'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# Set seeds for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Configure device IDs for multi-GPU training
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# Clear CUDA cache
torch.cuda.empty_cache()

# Train model
model = train_lightning_model(args, setting)
print(f'>>>>>>>training completed : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')

# Final cleanup
torch.cuda.empty_cache() 