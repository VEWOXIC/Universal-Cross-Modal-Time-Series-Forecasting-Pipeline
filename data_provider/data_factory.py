
from data_provider.data_loader import Universal_Dataset, Heterogeneous_Dataset
from torch.utils.data import DataLoader
import json, torch, yaml, os
from utils.tools import dotdict
from functools import partial
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
from tqdm import tqdm
from time import time

class Data_Provider(object):
    def __init__(self, args, buffer=False):
        self.args = args
        self.buffer = buffer
        self.batch_size = args.batch_size

        self.dataset_config = args.data_config

        self.id_info = json.load(open(os.path.join(self.dataset_config.root_path, self.dataset_config.id_info)))

        if self.dataset_config.id == 'all':
            self.id_list = self.id_info.keys()
        else:
            self.id_list = self.dataset_config.id
            # check if all the id in the list is in the id_info
            for id in self.dataset_config.id:
                assert id in self.id_info.keys(), "The id {} is not in the id_info".format(id)

        self.formatter = self.dataset_config.get('formatter', 'id_{i}.parquet')
        self.spliter = self.get_spliter()

        if buffer:
            self.data_buffer = data_buffer()

        if args.data_config.hetero_info is not None:
            if args.data_config.hetero_info['root_path'] is None:
                args.data_config.hetero_info['root_path'] = args.data_config.root_path
            
            self.hetero_dataset = Heterogeneous_Dataset(root_path=args.data_config.hetero_info['root_path'], formatter=args.data_config.hetero_info['formatter'], id_info=self.id_info, sampling_rate=args.data_config.hetero_info['sampling_rate'], matching=args.data_config.hetero_info['matching'], output_format=args.data_config.hetero_info['input_format'])

    def get_spliter(self):
        if self.dataset_config.spliter == 'timestamp':
            spliter = partial(timestamp_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len, timestamp_col=self.dataset_config.timestamp_col)
        elif self.dataset_config.spliter == 'ratio':
            spliter = partial(ratio_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len)
        else:
            print('no split method specified, use ratio of 7:1:2 as default')
            spliter = partial(ratio_spliter, split=(7,1,2), seq_len=self.args.input_len)
        return spliter
    
    def get_train(self):
        data_set, data_loader=self.get_loader('train', True, True, True)
        self.train_dataset = data_set
        return data_set, data_loader
    
    def get_val(self):
        data_set, data_loader=self.get_loader('val', True, True, True)
        self.val_dataset = data_set
        return data_set, data_loader
    
    def get_test(self):
        data_set, data_loader=self.get_loader('test', False, False, False)
        self.test_dataset = data_set

        return data_set, data_loader

    def get_loader(self, flag, shuffle, drop_last, concat=False):
        datasets = {}
        for i in tqdm(self.id_list, desc=f"Loading {flag} datasets"):
            if self.args.data_config.hetero_info is not None:
                get_hetero_data = self.hetero_dataset.init_hetero_data(i)
            else:
                get_hetero_data = None

            data_path = self.formatter.format(i=i)
            dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path, flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, target=self.dataset_config.target, scale=self.args.scale, data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data) 
            datasets[i] = dataset

        if concat:
            data_set = torch.utils.data.ConcatDataset([datasets[i] for i in datasets.keys()])
            data_loader = DataLoader(data_set,
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers)
            return data_set, data_loader
        else:
            data_loader = {}
            for i in datasets.keys():
                data_loader[i] = DataLoader(datasets[i],
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers
                                    )

            return datasets, data_loader

        
        


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
