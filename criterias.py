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


def evaluate_full_dataset(loader, model, config, device, indexes, channel_wise): # MODIFIED: Added channel_wise parameter
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates the true MSE and MAE over the entire dataset, with an option for channel-wise evaluation.
    """
    total_mse, total_mae, num_samples = 0.0, 0.0, 0
    channel_mse, channel_mae, channel_counts = None, None, None

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel = iter_data

            batch_x = torch.tensor(batch_x).to(device)
            batch_y = torch.tensor(batch_y).to(device)
            
            if config.task == 'TSF':
                prediction = model(x=batch_x)
            elif config.task == 'TGTSF':
                x_hetero = torch.tensor(x_hetero).to(device)
                y_hetero = torch.tensor(y_hetero).to(device)
                hetero_channel = torch.tensor(hetero_channel).to(device)
                prediction = model(x=batch_x, historical_events=x_hetero, news=y_hetero, channel_description=hetero_channel)
            else:
                # todo
                pass
            
            prediction = prediction[:, -config.output_len:, :]

            if channel_wise:
                if channel_mse is None:
                    C = prediction.shape[2]
                    channel_mse = [0.0] * C
                    channel_mae = [0.0] * C
                    channel_counts = [0] * C
                for k in range(prediction.shape[2]):
                    mse_loss = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse[k] += mse_loss.item() * batch_y.size(0)
                    channel_mae[k] += mae_loss.item() * batch_y.size(0)
                    channel_counts[k] += batch_y.size(0)
            else:
                mse_loss = torch.nn.MSELoss()(prediction, batch_y)
                mae_loss = torch.nn.L1Loss()(prediction, batch_y)
                total_mae += mae_loss.item() * batch_y.size(0)
                total_mse += mse_loss.item() * batch_y.size(0)
                num_samples += batch_y.size(0)

    if channel_wise:
        return channel_mse, channel_mae, channel_counts
    else:
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
    parser.add_argument('--channel_wise', type=bool, default=False, help='Channel wise testing')
    
    args = parser.parse_args()

    # --- Find and Load Checkpoint ---
    ckpt_pattern = f'_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
    
    if args.version == 'latest':
        ckpt_paths = [os.path.join(args.checkpoint_base, d) for d in os.listdir(args.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif args.version == 'oldest':
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
    
    config.gpu = args.device
    config.num_workers = 0
    config.task = args.task
    config.batch_size = 1 if args.filtered_samples is not None else args.batch_size
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # --- Initialize and Load Model ---
    model = model_init(config.model, config.model_config, config).to(device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {ckpt_path}")
    
    ckpt_file_path = ckpt_file[0]
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device)

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
    if args.channel_wise:
        all_mae = {}
        all_mse = {}
        all_sample_num = {}
    else:
        all_mae = 0.0
        all_mse = 0.0
        all_sample_num = 0

    if args.filtered_samples is not None:
        filtered_samples = json.load(open(args.filtered_samples))
        print(f"[Info] Using filtered samples from: {args.filtered_samples}")
    
    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if args.filtered_samples is not None:
            indexes = filtered_samples.get(name, []) # Use .get for safety
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = evaluate_full_dataset(loader, model, config, device, indexes, args.channel_wise)

        if args.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if channel_mse is None or sum(channel_counts) == 0:
                print(f"-> No valid samples found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, Channel-wise MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': Overall Channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, Overall Channel MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")

                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
            else:
                print(f"-> No valid samples found in '{name}'")

    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")

    if args.channel_wise:
        # Check if there are any results to summarize
        if not all_mse:
            print("-> No results to summarize.")
        else:
            sum_mse = [sum(m) for m in zip(*all_mse.values())]
            sum_mae = [sum(m) for m in zip(*all_mae.values())]
            sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
            overall_mse_list = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
            overall_mae_list = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
            print(f"-> Overall Results (All Subsets): Channel-wise MSE = {overall_mse_list}, Channel-wise MAE = {overall_mae_list}")
            overall_mse = sum(overall_mse_list) / len(overall_mse_list) if overall_mse_list else 0
            overall_mae = sum(overall_mae_list) / len(overall_mae_list) if overall_mae_list else 0
            print(f"-> Overall Results (All Subsets): MSE = {overall_mse:.7f}, MAE = {overall_mae:.7f}")

    else:
        if all_sample_num > 0:
            print(f"-> Overall Results (All Subsets): MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
        else:
            print("-> No samples were processed.")
    
    print("="*50)


if __name__ == '__main__':
    main()
    sys.exit(0)
