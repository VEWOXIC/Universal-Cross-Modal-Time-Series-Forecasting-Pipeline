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


def get_reasoning_samples(lossdf):

    lossdf_ok = lossdf[lossdf.loss_mutual > -10]
    lossdf_pos = lossdf_ok[lossdf_ok.loss_mutual > 0]
    lossdf_neg = lossdf_ok[lossdf_ok.loss_mutual < 0]
    prob_pos = lossdf_pos['loss_mutual']
    prob_neg = lossdf_neg['loss_mutual'].abs()
    sample_num_pos = int(len(lossdf_pos) * 0.1)
    sample_num_neg = int(len(lossdf_neg) * 0.1)


    while True:

        weights_pos = prob_pos.tolist()
        weights_neg = prob_neg.tolist()

        # Use random.choices for sampling
        sample_pos = random.choices(lossdf_pos.index.tolist(), weights=weights_pos, k=sample_num_pos)
        sample_neg = random.choices(lossdf_neg.index.tolist(), weights=weights_neg, k=sample_num_neg)
        break

    samples = np.concatenate((sample_pos, sample_neg))
    return samples.tolist()

def get_lossdf(dataset, model_TST, model_TGTSF, stride, config):
    losslist = {}
    for sample_num in tqdm(range(0, len(dataset), stride), desc="Processing samples"):
        with torch.no_grad():
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = dataset[sample_num]

            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(config.device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(config.device)
            batch_y_hetero = torch.tensor(batch_y_hetero).unsqueeze(0).float().to(config.device)
            hetero_channel = torch.tensor(hetero_channel).unsqueeze(0).float().to(config.device)

            output_TST = model_TST(x=batch_x)
            output_TST = output_TST[:, -config.output_len:, :]

            output_TGTSF = model_TGTSF(x=batch_x, historical_events =batch_x_hetero, news = batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)
            output_TGTSF = output_TGTSF[:, -config.output_len:, :]

            # calculate the loss
            loss_TST = torch.nn.MSELoss()(output_TST, batch_y)
            loss_TGTSF = torch.nn.MSELoss()(output_TGTSF, batch_y)

        losslist[sample_num] = {
            'loss_TST': loss_TST.item(),
            'loss_TGTSF': loss_TGTSF.item(),
            'loss_mutual': (loss_TST.item() - loss_TGTSF.item()) / loss_TST.item()
        }
    lossdf = pd.DataFrame.from_dict(losslist, orient='index')
    return lossdf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter reasoning samples")
    parser.add_argument('--data', type=str, required=True, help="Dataset name (e.g., 'solar')")
    parser.add_argument('--ahead', type=str, required=True, help="Prediction horizon (e.g., 'day')")
    parser.add_argument('--stride', type=int, required=True, help="Prediction horizon (e.g., 'day')")
    args = parser.parse_args()

    data = args.data
    ahead = args.ahead
    stride = int(args.stride)

    model = 'PatchTST'
    version = 'latest'
    ckpt_base = './checkpoints/'+data

    ckpt_id = f'_{model}_{data}_{ahead}_ahead'

    if version == 'latest':
        # find all the path that end with the ckpt_id
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if i.endswith(ckpt_id)]
        # the path is in format of yyyy-mm-dd{ckpt_id}, now find the latest one
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    else:
        ckpt_path = version + ckpt_id

    config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.batch_size = 1

    model_TST = model_init(config.model, config.model_config, config).to(config.device)
    # load the model
    model_TST.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth'))) 
    model_TST.eval()

    ################################

    model = 'TGTSF'
    version = 'latest'#'04-16-0316'#'latest'
    ckpt_base = './checkpoints/'+data

    ckpt_id = f'_{model}_{data}_{ahead}_ahead'

    if version == 'latest':
        # find all the path that end with the ckpt_id
        ckpt_paths = [os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        # the path is in format of yyyy-mm-dd{ckpt_id}, now find the latest one
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(ckpt_base, ckpt_path)

    config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.batch_size = 1
    print(ckpt_path)

    model_TGTSF = model_init(config.model, config.model_config, config).to(config.device)
    # load the model
    ckpt = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))[0]
    checkpoint = torch.load(ckpt)

    # Fix the state_dict by removing the "model." prefix
    if ckpt.endswith('.ckpt'):
        
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    # Load the fixed state_dict into your model
    model_TGTSF.load_state_dict(state_dict)
    model_TGTSF.eval()

    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    print(fullsets.keys())

    try:
        existing = os.path.join(f'./{data}_sample_{ahead}.json')
        existing = json.load(open(existing))
        existing = existing.keys()
    except:
        existing = []

    # set the seed

    np.random.seed(114514)
    random.seed(114514)

    sample_dict = {}
    for i in fullsets.keys():
        if i in existing:
            
            continue
        dataset = fullsets[i]
        print(i)
        lossdf = get_lossdf(dataset, model_TST, model_TGTSF, stride, config)
        lossdf.to_csv(os.path.join(ckpt_path, f'lossdf_{i}.csv'))
        try:
            samples = get_reasoning_samples(lossdf)
        except:
            print(f'error on {i}')
            continue
        print(samples)
        sample_dict[i] = samples
        with open(os.path.join(f'./{data}_sample_{ahead}.json'), 'w') as f:
            json.dump(sample_dict, f)