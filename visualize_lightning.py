import torch
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import glob
import matplotlib.pyplot as plt

from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def plot_prediction(indate, input_data, outdate, output_data, prediction_data, subset_name, sample_id, img_path):
    
    plt.figure(figsize=(15, 7), dpi=300)
    plt.plot(indate, input_data, label='Input History')
    plt.plot(outdate, output_data, label='Ground Truth')
    plt.plot(outdate, prediction_data, label='Prediction', linestyle='--')
    plt.title(f'Prediction Visualization for Subset: {subset_name}, Sample: {sample_id}')
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
    plt.close()
    print(f"Prediction plot saved to {img_path}")


def run_visualization(args, model, config, fullsets):
    
    subset_name = args.vis_subset
    sample_id = args.vis_sample_id

    print(f"[Info] Running visualization for subset '{subset_name}', sample ID {sample_id}")

    if subset_name not in fullsets:
        raise KeyError(f"Subset '{subset_name}' not found. Available subsets: {list(fullsets.keys())}")
    
    dataset_to_vis = fullsets[subset_name]

    if not (0 <= sample_id < len(dataset_to_vis)):
        raise IndexError(f"Sample ID {sample_id} is out of bounds for subset '{subset_name}' which has {len(dataset_to_vis)} samples.")

    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = dataset_to_vis[sample_id]

    input_tensor = torch.tensor(seq_x).to(config.device).float().unsqueeze(0)
    y_hetero = torch.tensor(y_hetero).to(config.device).float().unsqueeze(0)
    hetero_channel = torch.tensor(hetero_channel).to(config.device).float().unsqueeze(0)

    with torch.no_grad():
        prediction_tensor = model(x=input_tensor) if args.task == 'TSF' else model(x=input_tensor, news=y_hetero, channel_description=hetero_channel)
        prediction_tensor = prediction_tensor[:, -config.output_len:, :]

    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S', errors='coerce')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S', errors='coerce')

    input_np = input_tensor.cpu().numpy().squeeze()
    output_np = np.array(seq_y).squeeze()
    prediction_np = prediction_tensor.cpu().numpy().squeeze()
    
    save_path = os.path.join(args.vis_save_path, args.task, f"{args.fig_name}_subset-{subset_name}_sample-{sample_id}.png")

    plot_prediction(
        indate=indate_dt,
        input_data=input_np,
        outdate=outdate_dt,
        output_data=output_np,
        prediction_data=prediction_np,
        subset_name=subset_name,
        sample_id=sample_id,
        img_path=save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Model Testing and Visualization")
    
    parser.add_argument('--data', type=str, default="California_ISO", help="Dataset name")
    parser.add_argument('--model', type=str, default="TGTSF", help="Model name (e.g., 'PatchTST')")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=360, help="Input length")
    parser.add_argument('--output_len', type=int, default=24, help="Prediction horizon")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--device', type=str, default="cuda:4" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument('--task', type=str, default='TGTSF', choices=['TSF', 'TGTSF'], help="Task type: TSF or TGTSF")
    
    parser.add_argument('--vis_subset', type=str, default='co2_Biogas_CO2', help="Name of the data subset to visualize from (e.g., 'test', 'val').")
    parser.add_argument('--vis_sample_id', type=int, default=0, help="The index of the sample to visualize within the subset.")
    parser.add_argument('--vis_save_path', type=str, default='./imgs', help="Directory to save visualization images.")
    parser.add_argument('--fig_name', type=str, default='fig.png', help="Name of the figure file to save.")

    args = parser.parse_args()

    data = args.data
    model = args.model
    version = args.version
    input_len = args.input_len
    output_len = args.output_len
    ckpt_base = args.checkpoint_base

    ckpt_id = f'_{model}_{data}_{output_len}_{input_len}'

    if version in ['latest', 'newest']:
        ckpt_paths = [i for i in os.listdir(ckpt_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
        args.fig_name = ckpt_path
        ckpt_path = os.path.join(ckpt_base, ckpt_path)
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
    config.batch_size = 1  # Remain batch size = 1
    
    config.model = args.model
    config.data = args.data

    print(f"[Info] Running on device: {config.device}")

    model = model_init(config.model, config.model_config, config).to(config.device)
    
    ckpt_file_pattern = os.path.join(ckpt_path, 'checkpoint*')
    ckpt_files = glob.glob(ckpt_file_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path} with pattern {ckpt_file_pattern}")
    ckpt_file = ckpt_files[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=config.device)


    if 'state_dict' in checkpoint:
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f'[Info] Successfully loaded model: {config.model}')


    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    print(f'[Info] Found {len(fullsets)} datasets to test: {list(fullsets.keys())}')

    run_visualization(args, model, config, fullsets)
