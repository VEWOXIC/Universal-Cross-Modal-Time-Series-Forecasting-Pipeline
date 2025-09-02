
from data_provider.data_loader import Universal_Dataset, Heterogeneous_Dataset
from torch.utils.data import DataLoader
import json, torch, yaml, os
from utils.tools import dotdict
from functools import partial
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
from tqdm import tqdm

class Data_Provider(object):
    """
    Central data management class for the Universal Cross-Modal Time Series Forecasting Pipeline.
    
    This class orchestrates data loading, preprocessing, and provides unified access to training,
    validation, and test datasets. It supports both traditional time series data and heterogeneous
    cross-modal data sources (text, events, etc.) for enhanced forecasting capabilities.
    
    Args:
        args: Configuration object containing all data and model parameters including:
            - data_config: Dataset configuration with root_path, id_info, target, etc.
            - batch_size: Batch size for data loaders
            - input_len: Length of input sequences
            - output_len: Length of prediction sequences
            - scale: Whether to apply standardization
            - noise: Noise injection settings
            - num_workers: Number of workers for data loading
            - prefetch_factor: Prefetch factor for data loaders
        buffer (bool): Whether to enable data buffering for improved performance.
            When True, loaded data files are cached in memory to avoid repeated I/O.
    
    Attributes:
        id_list: List of dataset IDs to process
        formatter: String formatter for dataset file naming (e.g., 'id_{i}.parquet')
        spliter: Function for splitting data into train/val/test sets
        data_buffer: Optional data buffer for caching
        hetero_dataset: Heterogeneous dataset handler for cross-modal data
    
    Example:
        ```python
        data_provider = Data_Provider(args, buffer=True)
        train_loader = data_provider.get_train(return_type='loader')
        val_loader = data_provider.get_val(return_type='loader')
        ```
    """
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
        else:
            self.data_buffer = None

        if args.data_config.hetero_info is not None:
            hetero_info = dotdict(args.data_config.hetero_info)
            if hetero_info.root_path is None:
                hetero_info.root_path = args.data_config.root_path
            
            self.hetero_dataset = Heterogeneous_Dataset(root_path=hetero_info.root_path, 
                                                        formatter=hetero_info.formatter, 
                                                        id_info=self.id_info, 
                                                        matching=hetero_info.matching, 
                                                        output_format=hetero_info.input_format, 
                                                        static_path=hetero_info.static_path, 
                                                        timezone=self.dataset_config.time_zone, 
                                                        noise=self.args.noise, 
                                                        hetero_type=hetero_info.hetero_type, 
                                                        id_list=self.id_list, 
                                                        postemb=hetero_info.postemb, 
                                                        postemb_model=hetero_info.postemb_model, 
                                                        postemb_max_len=hetero_info.postemb_max_len, 
                                                        postemb_d=hetero_info.postemb_d, 
                                                        postemb_batch_size=hetero_info.postemb_batch_size, 
                                                        postemb_handle_downtime=hetero_info.postemb_handle_downtime, 
                                                        device=self.args.gpu if self.args.use_gpu else 'cpu')

    def get_spliter(self):
        """
        Creates and returns a data splitting function based on configuration.
        
        Supports two splitting strategies:
        - 'timestamp': Split data based on specific timestamp boundaries
        - 'ratio': Split data based on proportional ratios (e.g., 7:1:2 for train:val:test)
        
        Returns:
            callable: Configured splitting function that takes a DataFrame and returns
                     (train_data, val_data, test_data) tuple
        """
        if self.dataset_config.spliter == 'timestamp':
            spliter = partial(timestamp_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len, timestamp_col=self.dataset_config.timestamp_col)
        elif self.dataset_config.spliter == 'ratio':
            spliter = partial(ratio_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len)
        else:
            print('no split method specified, use ratio of 7:1:2 as default')
            spliter = partial(ratio_spliter, split=(7,1,2), seq_len=self.args.input_len)
        return spliter
    
    def get_train(self, return_type='loader'):
        """
        Creates and returns training data in the specified format.
        
        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only  
                - 'both': Returns tuple of (dataset, dataloader)
        
        Returns:
            Dataset/DataLoader/tuple: Training data in requested format
        """
        assert return_type in ['set', 'loader', 'both'], 'return type not supported, only support set, loader, both'
        self.train_dataset=self.get_datasets('train')
        if return_type == 'set':
            return self.train_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.train_dataset, True, True, True)
        else:
            return self.train_dataset, self.get_dataloader(self.train_dataset, True, True, True)
    
    def get_val(self, return_type='loader'):
        """
        Creates and returns validation data in the specified format.
        
        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only  
                - 'both': Returns tuple of (dataset, dataloader)
        
        Returns:
            Dataset/DataLoader/tuple: Validation data in requested format
        """
        self.val_dataset = self.get_datasets('val')
        if return_type == 'set':
            return self.val_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.val_dataset, True, False, True)
        else:
            return self.val_dataset, self.get_dataloader(self.val_dataset, True, False, True)

    
    def get_test(self, return_type='loader'):
        """
        Creates and returns test data in the specified format.
        
        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only  
                - 'both': Returns tuple of (dataset, dataloader)
        
        Returns:
            Dataset/DataLoader/tuple: Test data in requested format
        """
        self.test_dataset = self.get_datasets('test')
        if return_type == 'set':
            return self.test_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.test_dataset, False, False, False)
        else:
            return self.test_dataset, self.get_dataloader(self.test_dataset, False, False, False)

    def get_datasets(self, flag):
        """
        Creates Universal_Dataset instances for all configured data IDs.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
        
        Returns:
            dict: Dictionary mapping data IDs to their corresponding Universal_Dataset instances
        """
        datasets = {}
        for i in tqdm(self.id_list, desc=f"Loading {flag} datasets"):
            if self.args.data_config.hetero_info is not None:
                get_hetero_data = self.hetero_dataset.init_hetero_data(i)
            else:
                get_hetero_data = None

            data_path = self.formatter.format(i=i)
            dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path, 
                                        flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, 
                                        spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, 
                                        target=self.dataset_config.target, scale=self.args.scale, 
                                        data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data, preload_hetero=self.args.preload_hetero, 
                                        hetero_stride=self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1,
                                        task=self.args.model_config.task, custom_input=self.args.model_config.custom_input,
                                        timezone=self.dataset_config.time_zone, downsample=self.dataset_config.downsample)
            datasets[i] = dataset
        return datasets

    def get_dataloader(self, datasets, shuffle, drop_last, concat=False):
        """
        Creates PyTorch DataLoader instances from datasets.
        
        Args:
            datasets (dict): Dictionary of dataset instances mapped by ID
            shuffle (bool): Whether to shuffle data during loading
            drop_last (bool): Whether to drop the last incomplete batch
            concat (bool): Whether to concatenate all datasets into a single loader
                          or return separate loaders for each dataset
        
        Returns:
            DataLoader or dict: Single DataLoader if concat=True, 
                               dict of DataLoaders mapped by ID if concat=False
        """
        if concat:
            data_set = torch.utils.data.ConcatDataset([datasets[i] for i in datasets.keys()])
            data_loader = DataLoader(data_set,
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers,
                                    persistent_workers=(self.args.num_workers > 1),
                                    pin_memory=True,
                                    prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 1 else None,
                                    )
            return data_loader
        else:
            data_loader = {}
            for i in datasets.keys():
                data_loader[i] = DataLoader(datasets[i],
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers,
                                    persistent_workers=(self.args.num_workers > 1),
                                    pin_memory=True,
                                    prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 1 else None,
                                    )

            return data_loader


