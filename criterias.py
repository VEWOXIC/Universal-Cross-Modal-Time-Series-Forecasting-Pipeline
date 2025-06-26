import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
from tqdm import tqdm
from utils.metrics import MAE, MSE


def evaluate_single_sample(args, model, test_set, config):
    print(f"--- Perform on ID {args.data_id}, sample {args.sample_num} ---")
    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = test_set[args.data_id][args.sample_num]

    input_tensor = torch.tensor(seq_x).to(config.device).float().unsqueeze(0)
    output_tensor = torch.tensor(seq_y).to(config.device).float().unsqueeze(0)

    with torch.no_grad():
        prediction_tensor = model(input_tensor)
        prediction_tensor = prediction_tensor[:, -config.output_len:, :]

    mae, mse = MAE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy()), MSE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy())
    print(f"MAE for this sample: {mae:.4f}")
    print(f"MSE for this sample: {mse:.4f}")


def evaluate_all_samples(args, model, test_set, config):
    """
    Evaluate all samples in the specified data ID and calculate criterias.
    """
    total_mae = 0.0
    total_mse = 0.0
    sample_count = 0
    num_samples = len(test_set[args.data_id])

    for sample_num in tqdm(range(num_samples), desc=f"Evaluating data ID {args.data_id}", unit="sample"):
        seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = test_set[args.data_id][sample_num]

        input_tensor = torch.tensor(seq_x).to(config.device).float().unsqueeze(0)
        output_tensor = torch.tensor(seq_y).to(config.device).float().unsqueeze(0)

        with torch.no_grad():
            prediction_tensor = model(input_tensor)
            prediction_tensor = prediction_tensor[:, -config.output_len:, :]

        mae, mse = MAE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy()), MSE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy())
        total_mae += mae
        total_mse += mse
        sample_count += 1

    avg_mae = total_mae / sample_count
    avg_mse = total_mse / sample_count

    print(f"Average MAE for all samples: {avg_mae:.4f}")
    print(f"Average MSE for all samples: {avg_mse:.4f}")


def criterias_main(args):
    """
    Main criterias program for the script.
    """
    # --- Load config and checkpoint ---
    ckpt_path = os.path.join(args.ckpt_base, args.ckpt_id)
    print(f"Loading checkpoint from: {ckpt_path}")

    config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.batch_size = 1

    # --- Load data ---
    D = Data_Provider(config)
    test_set = D.get_test("set")
    train_set = D.get_train("set")
    val_set = D.get_val("set")
    print(len(train_set["1"]), len(val_set["1"]), len(test_set["1"]))

    # --- Initialize and load model ---
    model = model_init(config.model, config.model_config, config).to(config.device)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=config.device))
    model.eval()

    # Evaluate a single sample
    if args.evaluate_mode == 'single_sample':
        evaluate_single_sample(args, model, test_set, config)
    # Evaluate all samples in the specified data ID
    elif args.evaluate_mode == 'all_samples':
        evaluate_all_samples(args, model, test_set, config)


if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='TSF/TGTSF Evaluation')
    
    # --- config ---
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints')
    parser.add_argument('--ckpt_id', type=str, default='06-20-0955_DLinear_ETT_96_288', help='Checkpoint folder ID')
    parser.add_argument('--data_id', type=str, default='1', help='Data ID to visualize')
    parser.add_argument('--sample_num', type=int, default=100, help='The sample index to visualize')
    parser.add_argument('--evaluate_mode', type=str, default='all_samples', choices=['single_sample', 'all_samples'], help='Mode to evaluate: single sample or all samples in data ID')

    args = parser.parse_args()
    
    criterias_main(args)
    