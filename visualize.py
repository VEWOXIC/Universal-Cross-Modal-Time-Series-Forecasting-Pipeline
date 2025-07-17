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


def plot_prediction(indate, input_data, outdate, output_data, prediction_data, data_id, sample_id, img_path):
    """
    figure the prediction visualization
    """
    plt.figure(figsize=(15, 7), dpi=300)
    plt.plot(indate, input_data, label='Input History')
    plt.plot(outdate, output_data, label='Ground Truth')
    plt.plot(outdate, prediction_data, label='Prediction', linestyle='--')
    plt.title(f'Prediction Visualization for ID: {data_id}, Sample: {sample_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()

    img_dir = os.path.dirname(img_path)
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")
    
    plt.savefig(img_path)
    print(f"Prediction plot saved to {img_path}")


def visualize_main(args):
    """
    Main visualize program for the script.
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
    # print(len(test_set["1"]))

    # --- Initialize and load model ---
    model = model_init(config.model, config.model_config, config).to(config.device)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=config.device))
    model.eval()

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

    # --- Visualization ---
    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S')

    input_np = input_tensor.cpu().numpy().squeeze()
    output_np = output_tensor.cpu().numpy().squeeze()
    prediction_np = prediction_tensor.cpu().numpy().squeeze()

    plot_prediction(indate_dt, input_np, outdate_dt, output_np, prediction_np, args.data_id, args.sample_id, 
                    os.path.join(args.img_path, args.task, f"{args.ckpt_id}_subset-{args.data_id}_sample-{args.sample_id}.png"))


if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='TSF/TGTSF Visualization')
    
    # --- config ---
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints')
    parser.add_argument('--ckpt_id', type=str, default='06-27-1728_DLinear_ETT_96_720', help='Checkpoint folder ID')
    parser.add_argument('--data_id', type=str, default='1', help='Data ID to visualize')
    parser.add_argument('--sample_id', type=int, default=10, help='The sample index to visualize')
    parser.add_argument('--img_path', type=str, default='./imgs', help='Path to save the prediction visualization image')
    parser.add_argument('--task', type=str, default='TSF', choices=['TSF', 'TGTSF'], help='Task type: TSF or TGTSF')

    args = parser.parse_args()
    
    visualize_main(args)
    