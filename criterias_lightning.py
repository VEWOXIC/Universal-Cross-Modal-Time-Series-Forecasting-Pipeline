import torch
import pandas as pd
import numpy as np
import os
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
import json
import yaml
from tqdm import tqdm
import argparse
import glob
import sys


def run_test(loader, model, config, device, indexes, channel_wise):
    
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0

    channel_mse = None
    channel_mae = None
    channel_counts = None

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel = iter_data
            batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel = model.move_to_device(batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel, device)

            prediction = model(x=batch_x) if config.task == 'TSF' else model(x=batch_x, historical_events=x_hetero, news=y_hetero, channel_description=hetero_channel)
            prediction = prediction[:, -config.output_len:, :]  # [B, L, C]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Model Testing")
    parser.add_argument('--data', type=str, default="NYC_traffic_speed", help="Dataset name")
    parser.add_argument('--data_config', type=str, default=None, help='Data config if using another dataset/splitting method, else None')
    parser.add_argument('--baseline_model', type=str, default="DLinear", help="Model name (e.g., 'PatchTST')")
    parser.add_argument('--task', type=str, default="TSF", choices=["TSF", "TGTSF"], help="Task type")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest' 'oldest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=4320, help="Input length")
    parser.add_argument('--output_len', type=int, default=8640, help="Prediction horizon")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for testing")
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    parser.add_argument('--filtered_samples', type=str, default=None, help='filtered samples for testing')
    parser.add_argument('--channel_wise', type=bool, default=False, help='Channel wise testing')

    args = parser.parse_args()

    data = args.data
    baseline_model = args.baseline_model
    version = args.version
    input_len = args.input_len
    output_len = args.output_len
    ckpt_base = args.checkpoint_base

    ckpt_id = f'_{baseline_model}_{data}_{output_len}_{input_len}_pl'

    if version == 'latest':
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif args.version == 'oldest':
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(ckpt_base, version + ckpt_id)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

    print(f'[Info] Using checkpoint path: {ckpt_path}')


    results_save_dir = ckpt_path

    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"args.json not found in {ckpt_path}")
    
    config = dotdict(json.load(open(config_path)))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config) if args.data_config is None else dotdict(yaml.safe_load(open(args.data_config, 'r')))
    config.devices = args.device
    config.num_workers = 0
    config.batch_size = 1 if args.filtered_samples is not None else args.batch_size  # Must remain batch size = 1 for filtered testing
    config.task = args.task

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    model = model_init(config.model, config.model_config, config).to(device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path}")
    ckpt_file = ckpt_file[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)


    if ckpt_file.endswith('.ckpt'):
        if args.baseline_model == "PatchTST":
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()} 
        else:
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()} 

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f'[Info] Successfully loaded model: {config.model}')

    # Prepare datasets
    id_data = Data_Provider(config)
    fullloader = id_data.get_test('loader')
    print(f'[Info] Found {len(fullloader)} datasets to test: {list(fullloader.keys())}')

    # Handle filtered samples if provided
    if args.filtered_samples is not None:
        filtered_samples = json.load(open(args.filtered_samples))
        print(f"[Info] Using filtered samples from: {args.filtered_samples}")

    all_mae = 0.0 if not args.channel_wise else {}
    all_mse = 0.0 if not args.channel_wise else {}
    all_sample_num = 0 if not args.channel_wise else {}

    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if args.filtered_samples is not None:
            indexes = filtered_samples[name]
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = run_test(loader, model, config, device, indexes, args.channel_wise)
        if args.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if sum(channel_counts) == 0:
                print(f"-> No index found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': All channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")
            else:
                print(f"-> No index found in '{name}'")
            
    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    if args.channel_wise:
        sum_mse = [sum(m) for m in zip(*all_mse.values())]
        sum_mae = [sum(m) for m in zip(*all_mae.values())]
        sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
        overall_mse = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
        overall_mae = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
        print(f"-> Results for all subsets: channel-wise MSE = {overall_mse}, channel-wise MAE = {overall_mae}")
        print(f"-> Results for all subsets: All channel MSE = {sum(overall_mse) / len(overall_mse):.7f}, MAE = {sum(overall_mae) / len(overall_mae):.7f}")
    else:
        print(f"-> Results for all subsets: MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
    print("="*50)

    sys.exit(0)
