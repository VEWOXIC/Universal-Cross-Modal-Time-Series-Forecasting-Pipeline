
from data_provider.data_loader import Universal_Dataset
from torch.utils.data import DataLoader
import json, torch, yaml, os
from utils.tools import dotdict
from functools import partial
from .data_helper import timestamp_spliter, ratio_spliter
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from time import time


def data_provider(args, flag, buffer=None):
    # import copy
    # args = copy.deepcopy(args)
    

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    elif flag == 'val':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    elif flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    dataset_config = args.data_config
    # dataset_config = dotdict(dataset_config)
    datasets = {}
    id_info = json.load(open(os.path.join(dataset_config.root_path, dataset_config.id_info)))

    if dataset_config.id == 'all':
        id_list = id_info.keys()
    else:
        id_list = dataset_config.id
        # check if all the id in the list is in the id_info
        for id in dataset_config.id:
            assert id in id_info.keys(), "The id {} is not in the id_info".format(id)

    if dataset_config.spliter == 'timestamp':
        spliter = partial(timestamp_spliter, split=dataset_config.split_info, seq_len=args.input_len, timestamp_col=dataset_config.timestamp_col)
    elif dataset_config.spliter == 'ratio':
        spliter = partial(ratio_spliter, split=dataset_config.split_info, seq_len=args.input_len)
    else:
        print('no split method specified, use ratio of 7:1:2 as default')
        spliter = partial(ratio_spliter, split=(7,1,2), seq_len=args.input_len)

    formatter = dataset_config.get('formatter', 'id_{i}.parquet')
    for i in tqdm(id_list, desc="Loading datasets"):
        data_path = formatter.format(i=i)
        dataset = Universal_Dataset(root_path=dataset_config.root_path, data_path=data_path, flag=flag, seq_len=args.input_len, pred_len=args.output_len, spliter=spliter, timestamp_col=dataset_config.timestamp_col, target=dataset_config.target, scale=args.scale, data_buffer=buffer) # keep num_workers to 1 for test set for ordering 
        datasets[i] = dataset

    if flag != 'test':
        
        data_set = torch.utils.data.ConcatDataset([datasets[i] for i in datasets.keys()])

        print('Dataset length: ', len(data_set))

        data_loader = DataLoader(data_set, 
                                batch_size=batch_size, 
                                shuffle=shuffle_flag, 
                                drop_last=drop_last,
                                num_workers=args.num_workers)
        return data_set, data_loader
    
    else:
        data_set = datasets
        data_loader = {}
        for i in datasets.keys():
            data_loader[i] = DataLoader(datasets[i], 
                                batch_size=batch_size, 
                                shuffle=shuffle_flag, 
                                drop_last=drop_last,
                                num_workers=args.num_workers
                                )

        return data_set, data_loader
