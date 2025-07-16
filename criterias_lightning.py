import torch
import pandas as pd
import numpy as np
import os
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
import json
from tqdm import tqdm
import argparse, glob


def run_test(dataset, model, config):
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0

    for sample_num in tqdm(range(len(dataset)), desc="Running tests"):
        with torch.no_grad():
            batch_x, batch_y, _, _, _, _, _, _, _, _ = dataset[sample_num]

            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(config.device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(config.device)

            prediction = model(x=batch_x)
            prediction = prediction[:, -config.output_len:, :]

            mse_loss = torch.nn.MSELoss()(prediction, batch_y)
            mae_loss = torch.nn.L1Loss()(prediction, batch_y)

        total_mae += mae_loss.item()
        total_mse += mse_loss.item()
        num_samples += batch_y.shape[0]

    avg_mse = total_mse / num_samples if num_samples > 0 else 0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0
    
    return avg_mse, avg_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Model Testing")
    parser.add_argument('--data', type=str, default="NYC_traffic_speed", help="Dataset name")
    parser.add_argument('--baseline_model', type=str, default="DLinear", help="Model name (e.g., 'PatchTST')")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest' or a specific date like '2023-10-26')", choices=['latest', 'newest'])
    parser.add_argument('--input_len', type=int, default=4320, help="Input length")
    parser.add_argument('--output_len', type=int, default=8640, help="Prediction horizon")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size used during training (for config)")
    parser.add_argument('--device', type=str, default="cuda:4" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    args = parser.parse_args()

    data = args.data
    baseline_model = args.baseline_model
    version = args.version
    input_len = args.input_len
    output_len = args.output_len
    ckpt_base = args.checkpoint_base

    ckpt_id = f'_{baseline_model}_{data}_{output_len}_{input_len}'

    if version in ['latest', 'newest']:
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(ckpt_base, ckpt_path)

    print(f'[Info] Using checkpoint path: {ckpt_path}')


    results_save_dir = ckpt_path

    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"args.json not found in {ckpt_path}")
    
    config = dotdict(json.load(open(config_path)))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)
    
    config.device = torch.device(args.device)
    config.batch_size = args.batch_size

    print(f"[Info] Running on device: {config.device}")


    model_TST = model_init(config.model, config.model_config, config).to(config.device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path}")
    ckpt_file = ckpt_file[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=config.device)


    if ckpt_file.endswith('.ckpt'):
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model_TST.load_state_dict(state_dict)
    model_TST.eval()

    print(f'[Info] Successfully loaded model: {config.model}')


    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    print(f'[Info] Found {len(fullsets)} datasets to test: {list(fullsets.keys())}')

    all_results = {}
    

    for name, dataset in fullsets.items():
        print(f"\n[Info] Testing on dataset: {name}")
        
        mean_mse, mean_mae = run_test(dataset, model_TST, config)
        
        all_results[name] = {'MSE': mean_mse, 'MAE': mean_mae}
        
        print(f"-> Results for '{name}': MSE = {mean_mse:.4f}, MAE = {mean_mae:.4f}")


    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    print("="*50)

    summary_df = pd.DataFrame.from_dict(all_results, orient='index')
    
    if not summary_df.empty:

        average_metrics = summary_df.mean()
        all_results['Average'] = {'MSE': average_metrics['MSE'], 'MAE': average_metrics['MAE']}
        
        summary_df.loc['Average'] = average_metrics
        print(summary_df.round(4))

        summary_filename = os.path.join(results_save_dir, f'summary_results_{data}_{baseline_model}.json')
        with open(summary_filename, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n[Info] Summary results saved to {summary_filename}")
    
    else:
        print("No datasets were tested.")

    print("="*50)
