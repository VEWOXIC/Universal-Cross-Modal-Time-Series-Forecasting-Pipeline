
import argparse
import os
import torch
from exp.exp_uni import Exp_uni
import random
import numpy as np
import time
import yaml
from utils.tools import dotdict
from utils.task import ahead_task_parser

parser = argparse.ArgumentParser(description='CWTFormer for Time Series Forecasting')

# model config
parser.add_argument('--model', type=str, default='FITS', help='model name')
parser.add_argument('--model_config', type=str, default='model_configs/FITS.yaml', help='model config')

# data loader
parser.add_argument('--data', type=str, default='solar', help='data name')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--data_config', type=str, default='./data_configs/data_profile.yaml', help='data profile')
parser.add_argument('--scale', type=bool, default=True, help='scale data')
parser.add_argument('--disable_buffer', default=False, action='store_true', help='disable data buffer')

# forecasting task
parser.add_argument('--ahead', type=str, default=None, help='day/week/month ahead forecasting')
parser.add_argument('--output_len', type=int, default=1000, help='output sequence length or "ntp" for next token prediction')
parser.add_argument('--input_len', type=int, default=1000, help='output sequence length or "ntp" for next token prediction')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=96, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()

# preload the yamls
with open(args.model_config, 'r') as f:
    model_configs = yaml.safe_load(f)
model_configs = dotdict(model_configs)
args.model_config = model_configs

with open(args.data_config, 'r') as f:
    data_configs = yaml.safe_load(f)
data_configs = dotdict(data_configs)
args.data_config = data_configs

# get current time
current_time = time.strftime('%m-%d-%H%M', time.localtime(time.time()))
# setting record of experiment

if args.ahead is not None:
    assert args.ahead in ['day', 'week', 'month'], 'ahead task not supported, or add your own parser'

    try:
        args.output_len, args.input_len = ahead_task_parser(args.ahead, data_configs.sampling_rate)
        setting = f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead'
    except:
        setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
        raise ValueError('sampling rate not found in data config, fall back to default, input output length')
else:
    setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

torch.cuda.empty_cache()

Exp = Exp_uni

exp = Exp(args)  # set experiments
print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.train(setting)

torch.cuda.empty_cache()