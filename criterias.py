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
    print(f"--- Perform on ID {args.data_id}, sample {args.sample_id} ---")
    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = test_set[args.data_id][args.sample_id]

    input_tensor = torch.tensor(seq_x).to(config.device).float().unsqueeze(0)
    output_tensor = torch.tensor(seq_y).to(config.device).float().unsqueeze(0)
    y_hetero = torch.tensor(y_hetero).to(config.device).float().unsqueeze(0)
    hetero_channel = torch.tensor(hetero_channel).to(config.device).float().unsqueeze(0)

    with torch.inference_mode():
        with torch.no_grad():
            if args.task == 'TSF':
                # For TSF, we only need seq_x
                prediction_tensor = model(x=input_tensor)
                prediction_tensor = prediction_tensor[:, -config.output_len:, :]
            elif args.task == 'TGTSF':
                # For TGTSF, we need to pass news and channel description
                prediction_tensor = model(x=input_tensor, news=y_hetero, channel_description=hetero_channel)
                prediction_tensor = prediction_tensor[:, -config.output_len:, :]
            else:
                raise ValueError("Task type must be either 'TSF' or 'TGTSF'.")

    mae, mse = MAE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy()), MSE(prediction_tensor.cpu().numpy(), output_tensor.cpu().numpy())
    print(f"MAE for this sample: {mae:.4f}")
    print(f"MSE for this sample: {mse:.4f}")


def evaluate_all_samples(args, model, test_loader, config):
    """
    Evaluate all samples using batch processing, calculating the true MSE over the entire dataset.
    """
    print(f"--- Evaluating all samples for data ID {args.data_id} ---")

    total_squared_error = 0.0
    total_elements = 0
    
    model.eval()  # Ensure model is in evaluation mode

    # test_loader is likely a dictionary, iterate through its values
    for info, loader in test_loader.items():
        # Reset counters for each part of the test set if needed, or aggregate them
        # Here I assume we want one final MSE for everything.
        # If you want per-info MSE, move the initializations inside this loop.
        
        with torch.no_grad(): # More modern and safer than torch.inference_mode() in some edge cases
            for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc=f"Evaluating on {info}"):
                
                # Unpack the iter data
                batch_x, batch_y, _, _, _, batch_y_hetero, _, _, _, hetero_channel = iter_data

                # Move tensors to the configured device
                input_tensor = batch_x.to(config.device).float()
                output_tensor = batch_y.to(config.device).float()

                prediction_tensor = None 
                if args.task == 'TSF':
                    prediction_tensor = model(x=input_tensor)
                elif args.task == 'TGTSF':
                    y_hetero_tensor = batch_y_hetero.to(config.device).float()
                    hetero_channel_tensor = hetero_channel.to(config.device).float()
                    prediction_tensor = model(x=input_tensor, news=y_hetero_tensor, channel_description=hetero_channel_tensor)
                else:
                    raise ValueError("Task type must be either 'TSF' or 'TGTSF'.")
                
                # 1. Calculate the sum of squared errors for the current batch
                #    (prediction - true)^2, then sum over all elements in the tensor
                batch_loss = torch.sum((prediction_tensor - output_tensor) ** 2)
                
                # 2. Accumulate the total squared error
                total_squared_error += batch_loss.item()
                
                # 3. Accumulate the total number of elements
                #    output_tensor.numel() gives the total number of elements (e.g., batch_size * seq_len * features)
                total_elements += output_tensor.numel()

    # 4. Calculate the final, true MSE after the loop
    final_mse = total_squared_error / total_elements

    print(f"Average MSE for all samples: {final_mse:.4f}")
    
    # You can return this value if needed
    return final_mse


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

    # --- Initialize and load model ---
    model = model_init(config.model, config.model_config, config).to(config.device)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=config.device))
    model.eval()

    # --- Load data ---
    D = Data_Provider(config)

    # Evaluate a single sample
    if args.evaluate_mode == 'single_sample':
        evaluate_single_sample(args, model, D.get_test("set"), config)
    # Evaluate all samples in the specified data ID
    elif args.evaluate_mode == 'all_samples':
        evaluate_all_samples(args, model, D.get_test("loader"), config)


if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='TSF/TGTSF Evaluation')
    
    # --- config ---
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints')
    parser.add_argument('--ckpt_id', type=str, default='06-27-1728_DLinear_ETT_96_720', help='Checkpoint folder ID')
    parser.add_argument('--data_id', type=str, default='1', help='Data ID for evaluation')
    parser.add_argument('--sample_id', type=int, default=0, help='The sample index for evaluation')
    parser.add_argument('--evaluate_mode', type=str, default='all_samples', choices=['single_sample', 'all_samples'], help='Mode to evaluate: single sample or all samples in data ID')
    parser.add_argument('--task', type=str, default='TSF', choices=['TSF', 'TGTSF'], help='Task type: TSF or TGTSF')

    args = parser.parse_args()
    
    criterias_main(args)
    