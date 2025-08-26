import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
import glob
import sys
import yaml
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
from utils.metrics import MAE, MSE


def evaluate_full_dataset(loader, model, config, device, indexes):
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates the true MSE and MAE over the entire dataset.
    """
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0
    
    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, _, y_hetero, _, _, _, hetero_channel = iter_data

            batch_x = torch.tensor(batch_x).to(device)
            batch_y = torch.tensor(batch_y).to(device)
            y_hetero = torch.tensor(y_hetero).to(device)
            hetero_channel = torch.tensor(hetero_channel).to(device)

            prediction = model(x=batch_x) if config.task == 'TSF' else model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            prediction = prediction[:, -config.output_len:, :]

            mse_loss = torch.nn.MSELoss()(prediction, batch_y)
            mae_loss = torch.nn.L1Loss()(prediction, batch_y)

            total_mae += mae_loss.item() * batch_y.size(0)
            total_mse += mse_loss.item() * batch_y.size(0)
            num_samples += batch_y.size(0)
    
    return total_mse, total_mae, num_samples


def main():
    """
    Main entry point for the evaluation script.
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model Evaluation')
    
    # --- Checkpoint and Model Config ---
    parser.add_argument('--model', type=str, default="DLinear", help="Model name (e.g., 'DLinear', 'PatchTST')")
    parser.add_argument('--data', type=str, default="ETTm1", help="Dataset name used for training (e.g., 'ETTm1')")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest' 'oldest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=360, help="Input sequence length")
    parser.add_argument('--output_len', type=int, default=24, help="Output sequence length (prediction horizon)")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for testing")
    parser.add_argument('--data_config', type=str, default=None, help="Path to the data configuration YAML file (optional)")
    parser.add_argument('--task', type=str, default="TSF", choices=["TSF", "TGTSF"], help="Task type: Time Series Forecasting or Text-Grounded TSF")
    parser.add_argument('--filtered_samples', type=str, default=None, help='Path to a JSON file containing filtered sample indexes for evaluation')
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    
    args = parser.parse_args()

    # --- Find and Load Checkpoint ---
    ckpt_pattern = f'_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
    
    if args.version == 'latest':
        # Find all matching checkpoint directories and sort them to get the latest one
        ckpt_paths = [os.path.join(args.checkpoint_base, d) for d in os.listdir(args.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif args.version == 'oldest':
        # Find all matching checkpoint directories and sort them to get the oldest one
        ckpt_paths = [os.path.join(args.checkpoint_base, d) for d in os.listdir(args.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(args.checkpoint_base, args.version + ckpt_pattern)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

    print(f"[Info] Using checkpoint path: {ckpt_path}")

    # --- Load Configuration from Checkpoint Folder ---
    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'args.json' not found in {ckpt_path}")
    
    config = dotdict(json.load(open(config_path)))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config) if args.data_config is None else dotdict(yaml.safe_load(open(args.data_config, 'r')))
    
    # Override config with runtime arguments
    config.gpu = args.device
    config.num_workers = 0
    config.task = args.task
    config.batch_size = 1 if args.filtered_samples is not None else args.batch_size  # Must remain batch size = 1 for filtered testing
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # --- Initialize and Load Model ---
    model = model_init(config.model, config.model_config, config).to(device)
    
    # Find the checkpoint file (e.g., checkpoint.pth, model.ckpt)
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {ckpt_path}")
    
    ckpt_file_path = ckpt_file[0]
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device)

    # Handle different checkpoint formats (e.g., from PyTorch Lightning)
    if ckpt_file_path.endswith('.ckpt') and 'state_dict' in checkpoint:
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {config.model}')

    # --- Load Data ---
    data_provider = Data_Provider(config)
    fullloader = data_provider.get_test("loader")

    # --- Run Evaluation ---
    
    all_mae = 0.0
    all_mse = 0.0
    all_sample_num = 0

    if args.filtered_samples is not None:
        filtered_samples = json.load(open(args.filtered_samples))
        print(f"[Info] Using filtered samples from: {args.filtered_samples}")
    
    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if args.filtered_samples is not None:
            indexes = filtered_samples[name]
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        total_mse, total_mae, num_samples = evaluate_full_dataset(loader, model, config, device, indexes)

        if num_samples > 0:
            avg_mse = total_mse / num_samples if num_samples > 0 else 0
            avg_mae = total_mae / num_samples if num_samples > 0 else 0
            print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")

            all_mse += total_mse
            all_mae += total_mae
            all_sample_num += num_samples
        
        else:
            print(f"-> No index found in '{name}'")


    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    print(f"-> Results for all subsets: MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
    print("="*50)


if __name__ == '__main__':
    main()
    sys.exit(0)
