import pandas as pd
import numpy as np
import os
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
import yaml, json
import argparse
import random
import traceback

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter reasoning samples")
    parser.add_argument('--data', type=str, default="Canada_photovoltaics_plants", help="Dataset name")
    parser.add_argument('--baseline_model', type=str, default="PatchTST", help="Model name")
    parser.add_argument('--version', type=str, default="latest", help="Model version")
    parser.add_argument('--input_len', type=int, default=360, help="Input length")
    parser.add_argument('--output_len', type=int, default=24, help="Prediction horizon")
    parser.add_argument('--type', type=str, default="ckpt", help="Type of model checkpoint")
    parser.add_argument('--sample_root', type=str, default='./sample_random_indexes', help="Root directory")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory")
    parser.add_argument('--sampling_rate', type=float, default=0.1, help="Sampling rate")
    parser.add_argument('--device', type=str, default='0', help="Device")
    args = parser.parse_args()

    data = args.data
    baseline_model = args.baseline_model
    version = args.version
    input_len = args.input_len
    output_len = args.output_len
    checkpoint_type = args.type
    ckpt_base = args.checkpoint_base

    ckpt_id = f'_{baseline_model}_{data}_{output_len}_{input_len}'

    # 找到 ckpt 路径，读取 args.json 里的配置来初始化 dataset
    if version == 'latest':
        ckpt_paths =[os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif version == 'oldest':
        ckpt_paths =[os.path.join(ckpt_base, i) for i in os.listdir(ckpt_base) if ckpt_id in i]
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(ckpt_base, ckpt_path)

    print(f'[Info] Using config from checkpoint path: {ckpt_path}')

    config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config)
    config.batch_size = 1
    config.devices = args.device

    # 初始化数据集
    id_data = Data_Provider(config)
    fullsets = id_data.get_test('set')
    
    # 打印所有的 key 检查到底有几个 subset
    subset_keys = list(fullsets.keys())
    print(f'[Info] Found {len(subset_keys)} subsets: {subset_keys}')

    if output_len == 24:
        ahead = 'day'
    elif output_len == 168:
        ahead = 'week'
    elif output_len == 12:
        ahead = 'hour'
    elif output_len == 144:
        ahead = 'half_a_day'
    else:
        ahead = 'none'

    os.makedirs(args.sample_root, exist_ok=True)
    save_path = os.path.join(args.sample_root, f'{data}_sample_{ahead}.json')

    # 固定随机种子保证结果可复现
    np.random.seed(114514)
    random.seed(114514)

    # 我们直接创建一个新的字典，每次运行都覆盖旧文件（因为随机抽取极快，不需要跳过已存在的）
    sample_dict = {}
    total_samples = 0
    stride = int(output_len / 2)

    for i in subset_keys:
        try:
            dataset = fullsets[i]
            print(f"[Info] Handling subset: {i}")

            # 获取数据集长度
            dataset_len = len(dataset)
            
            # 生成该 dataset 中所有有效的 index
            valid_indices = list(range(0, dataset_len, stride))
            
            # 计算需要抽取的数量
            sample_num = int(len(valid_indices) * args.sampling_rate)
            sample_num = min(sample_num, len(valid_indices)) 
            
            if sample_num > 0:
                # 无放回随机抽样
                samples = random.sample(valid_indices, sample_num)
            else:
                samples =[]

            print(f"[Info] Subset {i}: Randomly selected {len(samples)} samples out of {len(valid_indices)}")
            total_samples += len(samples)

            sample_dict[i] = samples
            
            # 实时保存写入 JSON
            with open(save_path, 'w') as f:
                json.dump(sample_dict, f)

        except Exception as e:
            # 加入 Try-except 避免某个 Subset 异常导致整个脚本终止，并打印具体的报错信息
            print(f"[Error] Failed to process subset {i}. Error details:")
            traceback.print_exc()
            continue

    print("\n" + "="*50)
    print("Final Statistics Summary")
    print("="*50)
    print(f"Total Subsets Processed and Saved: {len(sample_dict)}")
    print(f"Total Random Samples Across Processed Subsets: {total_samples}")
    print(f"Sampling Rate Used: {args.sampling_rate * 100}%")
    print(f"File Saved At: {save_path}")
    print("="*50)