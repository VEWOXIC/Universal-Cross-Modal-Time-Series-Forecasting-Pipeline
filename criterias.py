import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
import glob
import yaml
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
from utils.metrics import MAE, MSE


def evaluate_single_sample(args, model, dataset_set, config):
    """
    Evaluates a single, specified sample from a dataset.
    """
    data_id = args.data_id
    sample_id = args.sample_id

    print(f"--- Performing evaluation on Dataset ID '{data_id}', Sample Index {sample_id} ---")

    if data_id not in dataset_set:
        raise KeyError(f"Data ID '{data_id}' not found in the test sets. Available IDs: {list(dataset_set.keys())}")
    
    dataset = dataset_set[data_id]
    if sample_id >= len(dataset):
        raise IndexError(f"Sample ID {sample_id} is out of bounds for dataset '{data_id}' which has {len(dataset)} samples.")

    # Unpack the specific sample
    seq_x, seq_y, _, _, _, y_hetero, _, _, _, hetero_channel = dataset[sample_id]

    # Prepare tensors for the model
    input_tensor = torch.tensor(seq_x).unsqueeze(0).float().to(config.device)
    output_tensor = torch.tensor(seq_y).unsqueeze(0).float().to(config.device)
    
    with torch.no_grad():
        if args.task == 'TSF':
            prediction_tensor = model(x=input_tensor)
        elif args.task == 'TGTSF':
            y_hetero_tensor = torch.tensor(y_hetero).unsqueeze(0).float().to(config.device)
            hetero_channel_tensor = torch.tensor(hetero_channel).unsqueeze(0).float().to(config.device)
            prediction_tensor = model(x=input_tensor, news=y_hetero_tensor, channel_description=hetero_channel_tensor)
        else:
            raise ValueError("Task type must be either 'TSF' or 'TGTSF'.")

    # Ensure prediction tensor is sliced to match output length
    prediction_tensor = prediction_tensor[:, -config.output_len:, :]

    # Calculate metrics
    mae = MAE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy())
    mse = MSE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy())
    
    print(f"MAE for this sample: {mae:.4f}")
    print(f"MSE for this sample: {mse:.4f}")
    return mse, mae


def evaluate_full_dataset(dataset, model, config, indexes):
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates the true MSE and MAE over the entire dataset.
    """
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0

    if indexes is not None:
        dataset = [dataset[i] for i in indexes]
    
    for sample_num in tqdm(range(len(dataset)), desc="Running tests"):
        with torch.no_grad():
            batch_x, batch_y, _, _, _, y_hetero, _, _, _, hetero_channel = dataset[sample_num]

            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(config.device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(config.device)
            y_hetero = torch.tensor(y_hetero).unsqueeze(0).float().to(config.device)
            hetero_channel = torch.tensor(hetero_channel).unsqueeze(0).float().to(config.device)

            prediction = model(x=batch_x) if config.task == 'TSF' else model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            prediction = prediction[:, -config.output_len:, :]

            mse_loss = torch.nn.MSELoss()(prediction, batch_y)
            mae_loss = torch.nn.L1Loss()(prediction, batch_y)

        total_mae += mae_loss.item()
        total_mse += mse_loss.item()
        num_samples += batch_y.shape[0]

    avg_mse = total_mse / num_samples if num_samples > 0 else 0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0
    
    return avg_mse, avg_mae


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
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size = 1")
    parser.add_argument('--data_config', type=str, default=None, help="Path to the data configuration YAML file (optional)")

    # --- Task and Evaluation Mode ---
    parser.add_argument('--task', type=str, default="TSF", choices=["TSF", "TGTSF"], help="Task type: Time Series Forecasting or Text-Grounded TSF")
    parser.add_argument('--evaluate_mode', type=str, default='all_samples', choices=['single_sample', 'all_samples'], help='Evaluate a single sample or all samples')
    parser.add_argument('--filtered_samples', type=str, default=None, help='Path to a JSON file containing filtered sample indexes for evaluation')

    # --- Single Sample Specific Args ---
    parser.add_argument('--data_id', type=str, default='test', help="Dataset ID to use for single sample evaluation (e.g., 'test', 'val')")
    parser.add_argument('--sample_id', type=int, default=0, help='The sample index for single sample evaluation')

    # --- System Config ---
    parser.add_argument('--device', type=str, default="cuda:1" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    
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
        # Construct the path from the specified version
        ckpt_folder_name = args.version + ckpt_pattern
        ckpt_path = os.path.join(args.checkpoint_base, ckpt_folder_name)

    print(f"[Info] Using checkpoint path: {ckpt_path}")

    # --- Load Configuration from Checkpoint Folder ---
    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'args.json' not found in {ckpt_path}")
    
    config = dotdict(json.load(open(config_path)))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config) if args.data_config is None else dotdict(yaml.safe_load(open(args.data_config, 'r')))
    
    # Override config with runtime arguments
    config.device = torch.device(args.device)
    config.task = args.task
    config.batch_size = 1  # Must remain batch size = 1 for filtered testing
    
    print(f"[Info] Running on device: {config.device}")

    # --- Initialize and Load Model ---
    model = model_init(config.model, config.model_config, config).to(config.device)
    
    # Find the checkpoint file (e.g., checkpoint.pth, model.ckpt)
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {ckpt_path}")
    
    ckpt_file_path = ckpt_file[0]
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=config.device)

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
    fullsets = data_provider.get_test("set")

    # --- Run Evaluation ---
    if args.evaluate_mode == 'single_sample':
        evaluate_single_sample(args, model, fullsets, config)
    
    elif args.evaluate_mode == 'all_samples':
        all_results = {}

        if args.filtered_samples is not None:
            filtered_samples = json.load(open(args.filtered_samples))
            print(f"[Info] Using filtered samples from: {args.filtered_samples}")
        

        for name, dataset in fullsets.items():
            print(f"\n[Info] Testing on dataset: {name}")

            if args.filtered_samples is not None:
                indexes = filtered_samples[name]
                print(f"[Info] Using {len(indexes)} filtered samples for testing.")
                print(f"[Info] Sample indexes: {indexes}")
            else:
                indexes = None
                print("[Info] Using all samples for testing.")
            
            mean_mse, mean_mae = evaluate_full_dataset(dataset, model, config, indexes)
            
            if mean_mse != 0 and mean_mae != 0:
                all_results[name] = {'MSE': mean_mse, 'MAE': mean_mae}
                print(f"-> Results for '{name}': MSE = {mean_mse:.7f}, MAE = {mean_mae:.7f}")
            else:
                print(f"-> No index found in '{name}'")


        print("\n" + "="*50)
        print(" " * 15 + "Overall Test Summary")
        print("="*50)

        summary_df = pd.DataFrame.from_dict(all_results, orient='index')
        
        if not summary_df.empty:

            average_metrics = summary_df.mean()
            all_results['Average'] = {'MSE': average_metrics['MSE'], 'MAE': average_metrics['MAE']}
            
            summary_df.loc['Average'] = average_metrics
            print(summary_df.round(4))

            results_save_dir = ckpt_path

            summary_filename = os.path.join(results_save_dir, f'summary_results_{args.data}_{args.model}.json')
            with open(summary_filename, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"\n[Info] Summary results saved to {summary_filename}")
        
        else:
            print("No datasets were tested.")

        print("="*50)


if __name__ == '__main__':
    main()