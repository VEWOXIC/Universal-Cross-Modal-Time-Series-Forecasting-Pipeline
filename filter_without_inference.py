import torch
import pandas as pd
import numpy as np
import os
from models import model_init
from data_provider.data_factory import Data_Provider
import matplotlib.pyplot as plt
from utils.task import ahead_task_parser
from utils.tools import dotdict
import yaml, json
from tqdm import tqdm
import argparse, glob
import random


def get_reasoning_samples(lossdf, sampling_rate):

    lossdf_ok = lossdf[lossdf.loss_mutual > -10]
    lossdf_pos = lossdf_ok[lossdf_ok.loss_mutual > 0]
    lossdf_neg = lossdf_ok[lossdf_ok.loss_mutual < 0]
    
    prob_pos = lossdf_pos['loss_mutual']
    prob_neg = lossdf_neg['loss_mutual'].abs()
    
    sample_num_pos = int(len(lossdf_pos) * sampling_rate)
    sample_num_neg = int(len(lossdf_neg) * sampling_rate)
    
    sample_num_pos = min(sample_num_pos, len(lossdf_pos))
    sample_num_neg = min(sample_num_neg, len(lossdf_neg))
    
    if sample_num_pos > 0:
        sample_pos = lossdf_pos.sample(
            n=sample_num_pos, 
            weights='loss_mutual', 
            replace=False  # Without replacement
        ).index.tolist()
    else:
        sample_pos = []
    
    if sample_num_neg > 0:
        sample_neg = lossdf_neg.sample(
            n=sample_num_neg, 
            weights=prob_neg, 
            replace=False  # Without replacement
        ).index.tolist()
    else:
        sample_neg = []
    
    return sample_pos, sample_neg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter reasoning samples")
    parser.add_argument('--data', type=str, default="Canada_photovoltaics_plants", help="Dataset name (e.g., 'solar')")
    parser.add_argument('--baseline_model', type=str, default="PatchTST", help="Model name (e.g., 'PatchTST')")
    parser.add_argument('--version', type=str, default="latest", help="Model version (e.g., 'latest')", choices=['latest', 'newest'])
    parser.add_argument('--input_len', type=int, default=360, help="Input length (e.g., 360)")
    parser.add_argument('--output_len', type=int, default=24, help="Prediction horizon (e.g., 168)")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint (e.g., 'ckpt')")
    parser.add_argument('--sample_root', type=str, default='./sample_indexes', help="Root directory for saving samples")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--sampling_rate', type=float, default=0.1, help="Sampling rate for each subset")
    # parser.add_argument('--ahead', type=str, required=True, help="Prediction horizon (e.g., 'day')")
    args = parser.parse_args()

    data = args.data
    baseline_model = args.baseline_model
    version = args.version
    input_len = args.input_len
    output_len = args.output_len
    checkpoint_type = args.type
    ckpt_base = args.checkpoint_base

    TG_model = 'TGTSF'
    ckpt_id = f'_{TG_model}_{data}_{output_len}_{input_len}'

    if version == 'latest':
        # find all the path that end with the ckpt_id
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        # the path is in format of yyyy-mm-dd{ckpt_id}, now find the latest one
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(ckpt_base, ckpt_path)

    print(f'[Info] Using checkpoint path: {ckpt_path}')

    config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)

    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    print(f'[Info] fullset keys: {fullsets.keys()}')

    # set the seed

    np.random.seed(114514)  # pandas also use np.random if "random" in ".sample" is not set
    random.seed(114514)

    if output_len == 24:
        ahead = 'day'
    elif output_len == 168:
        ahead = 'week'
    else:
        ahead = 'none'

    try:
        existing = os.path.join(args.sample_root, f'{data}_sample_{ahead}.json')
        existing = json.load(open(existing))
        existing = existing.keys()
    except:
        existing = []

    sample_dict = {}
    total_pos_samples, total_neg_samples = 0, 0

    for i in fullsets.keys():
        if i in existing:
            continue

        dataset = fullsets[i]
        print(f"[Info] handling {i}")

        lossdf = pd.read_csv(os.path.join(ckpt_path, f'lossdf_{i}.csv'), index_col=0)

        try:
            sample_pos, sample_neg = get_reasoning_samples(lossdf, args.sampling_rate)
            
            print(f"[Info]: selected {len(sample_pos)} pos samples and {len(sample_neg)} neg samples insubset {i}")
            total_pos_samples += len(sample_pos)
            total_neg_samples += len(sample_neg)

            samples = sample_pos + sample_neg

        except:
            print(f'[Error] on {i}')
            continue

        print(f"[Info] generated: {samples}")

        sample_dict[i] = samples
        with open(os.path.join(args.sample_root, f'{data}_sample_{ahead}.json'), 'w') as f:
            json.dump(sample_dict, f)
        

    print("\n" + "="*50)
    print("Final Statistics Summary")
    print("="*50)
    print(f"Total Positive Candidates Across All Processed Subsets: {total_pos_samples}")
    print(f"Total Negative Candidates Across All Processed Subsets: {total_neg_samples}")
    print("="*50)