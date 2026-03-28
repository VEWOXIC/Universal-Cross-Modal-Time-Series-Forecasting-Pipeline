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


def plot_prediction(indate, input_data, outdate, output_data, prediction_data, dataset_name, model_name, subset_name, sample_id, channel_id, img_path):
    
    plt.figure(figsize=(15, 7), dpi=300)
    plt.plot(indate, input_data, label='Input History')
    plt.plot(outdate, output_data, label='Ground Truth')
    plt.plot(outdate, prediction_data, label='Prediction', linestyle='--')
    plt.title(f'Prediction Visualization for {dataset_name}: {subset_name}: channel {channel_id}, Model: {model_name}, Sample: {sample_id}')
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


def run_visualization(args, model, config, device, fullsets):
    
    subset_name = args.vis_subset
    sample_id = args.vis_sample_id

    print(f"[Info] Running visualization for subset '{subset_name}', sample ID {sample_id}")

    if subset_name not in fullsets:
        raise KeyError(f"Subset '{subset_name}' not found. Available subsets: {list(fullsets.keys())}")
    
    dataset_to_vis = fullsets[subset_name]

    if not (0 <= sample_id < len(dataset_to_vis)):
        raise IndexError(f"Sample ID {sample_id} is out of bounds for subset '{subset_name}' which has {len(dataset_to_vis)} samples.")

    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = dataset_to_vis[sample_id]

    input_tensor = torch.tensor(seq_x).to(device).float().unsqueeze(0)
    y_hetero = torch.tensor(y_hetero).to(device).float().unsqueeze(0)
    hetero_channel = torch.tensor(hetero_channel).to(device).float().unsqueeze(0)

    with torch.no_grad():
        prediction_tensor = model(x=input_tensor) if args.task == 'TSF' else model(x=input_tensor, news=y_hetero, channel_description=hetero_channel)
        prediction_tensor = prediction_tensor[:, -config.output_len:, :]

    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S', errors='coerce')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S', errors='coerce')

    input_np = input_tensor.cpu().numpy().squeeze()
    output_np = np.array(seq_y).squeeze()
    prediction_np = prediction_tensor.cpu().numpy().squeeze()
    
    save_path = os.path.join(args.vis_save_path, args.task, f"{args.fig_name}_subset-{subset_name}_sample-{sample_id}.png")

    input_np = input_np if args.channel_id == 'all' else input_np[:, int(args.channel_id)]
    output_np = output_np if args.channel_id == 'all' else output_np[:, int(args.channel_id)]
    prediction_np = prediction_np if args.channel_id == 'all' else prediction_np[:, int(args.channel_id)]

    plot_prediction(
        indate=indate_dt,
        input_data=input_np,
        outdate=outdate_dt,
        output_data=output_np,
        prediction_data=prediction_np,
        dataset_name=args.data,
        model_name=args.model,
        subset_name=subset_name,
        sample_id=sample_id,
        channel_id=args.channel_id,
        img_path=save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Model Testing and Visualization")
    
    parser.add_argument('--data', type=str, default="Germany_Renewable_Power_Grid", help="Dataset name")
    parser.add_argument('--model', type=str, default="PatchTST", help="Model name (e.g., 'PatchTST')")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=360, help="Input length")
    parser.add_argument('--output_len', type=int, default=168, help="Prediction horizon")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--checkpoint_file', type=str, default="best", choices=["last", "best"], help="Specific checkpoint file to load (optional)")
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    parser.add_argument('--task', type=str, default='TSF', choices=['TSF', 'TGTSF'], help="Task type: TSF or TGTSF")
    
    parser.add_argument('--channel_id', type=str, default="all", help="'all' for all channels, or a specific channel number to visualize")
    parser.add_argument('--vis_subset', type=str, default='solar_TransnetBW', help="Name of the data subset to visualize from (e.g., 'test', 'val').")
    parser.add_argument('--vis_sample_id', type=int, default=1176, help="The index of the sample to visualize within the subset.")
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
    config.model = args.model
    config.data = args.data
    config.batch_size = 1  # Remain batch size = 1
    devices = args.device

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    model = model_init(config.model, config.model_config, config).to(device)
    
    ckpt_file_pattern = os.path.join(ckpt_path, 'checkpoint*') if args.checkpoint_file == "best" else os.path.join(ckpt_path, 'last*')
    
    ckpt_files = glob.glob(ckpt_file_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path} with pattern {ckpt_file_pattern}")
    ckpt_file = ckpt_files[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)


    if 'state_dict' in checkpoint:
        if args.task == 'TGTSF':
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
        elif args.task == 'TSF':
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items() if 'news' not in key}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f'[Info] Successfully loaded model: {config.model}')


    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    print(f'[Info] Found {len(fullsets)} datasets to test: {list(fullsets.keys())}')

    run_visualization(args, model, config, device, fullsets)
