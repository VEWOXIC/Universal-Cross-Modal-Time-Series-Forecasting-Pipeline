import pandas as pd
import os, torch
# import the default collate function
from torch.utils.data.dataloader import default_collate

class data_buffer():
    def __init__(self):
        self.buffer = {}

    def __call__(self, file_path, force_reload=False):
        if force_reload:
            print(f'Force reloading data from {file_path}')
            if file_path.endswith('.csv'):
                df_raw = pd.read_csv(file_path)
                self.buffer[file_path] = df_raw.copy()
            elif file_path.endswith('.parquet'):
                df_raw = pd.read_parquet(file_path)
                self.buffer[file_path] = df_raw.copy()
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
            return df_raw
        
        if file_path in self.buffer.keys():
            return self.buffer[file_path].copy()
        else:
            if file_path.endswith('.csv'):
                df_raw = pd.read_csv(file_path)
                self.buffer[file_path] = df_raw.copy()
            elif file_path.endswith('.parquet'):
                df_raw = pd.read_parquet(file_path)
                self.buffer[file_path] = df_raw.copy()
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
            print(f'Add data {file_path} to buffer')
            return df_raw
    def clear(self):
        self.buffer = {}
        print('Buffer cleared')
        

def ratio_spliter(split=(7,1,2),seq_len=0, df=None):
    # split mark
    train_split = split[0] / sum(split)
    val_split = split[1] / sum(split) + train_split    # [----train----] train_split [----val----] val_split [----test----]
    # check if split is string it should be in format "x:y:z"
    if isinstance(split, str):
        assert len(split.split(':')) == 3, "Split should be in format 'x:y:z'"
        split = [int(x) for x in split.split(':')]
    elif isinstance(split, tuple) or isinstance(split, list):
        assert len(split) == 3, "Split should be in format 'x:y:z', or list with length of 3"
    else:
        raise ValueError("Split should be in format 'x:y:z', or list with length of 3")
    
    raw_data_len = len(df)

    train_split = int(raw_data_len * train_split)
    val_split = int(raw_data_len * val_split)
    
    train_data = df[0:train_split]
    val_data = df[train_split-seq_len:val_split]
    test_data = df[val_split-seq_len:]

    return train_data, val_data, test_data

def timestamp_spliter(split = ['2020-01-01', '2020-02-01'], seq_len=0, df=None, timestamp_col='timestamp'):
    assert len(split) == 2, "Split should be a list of two strings of time stamps"
    if isinstance(split[0], str):
        split = [pd.to_datetime(x) for x in split]
    else:
        raise ValueError("Split should be a list of strings")
    train_data = df[df[timestamp_col] < split[0]]
    val_data = df[(df[timestamp_col] >= split[0]) & (df[timestamp_col] < split[1])]
    test_data = df[df[timestamp_col] >= split[1]]
    
    return train_data, val_data, test_data

def instance_variable_filter(seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, filter):
    """
    Filter the item based on the filter
    Args:
        item: dict
        filter: dict
    Returns:
        item: dict
    """
    if filter == 'all':
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
    if filter == 'TSF':
        return seq_x, seq_y
    if filter == 'TGTSF':
        return seq_x, seq_y, y_hetero, hetero_channel
    if isinstance(filter, list):
        # assert all the item in the list is in the [seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel]
        assert all([x in ['seq_x', 'seq_y', 'x_time', 'y_time', 'x_hetero', 'y_hetero', 'hetero_x_time', 'hetero_y_time', 'hetero_general', 'hetero_channel'] for x in filter]), "The filter should be a list of the item you want to keep, including [seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel]"

        return [eval(x) for x in filter]
    
def dict_collate_fn(batch):
    keys = batch[0].keys()
    batch = {key: [d[key] for d in batch] for key in keys}
    # convert to tensor
    batch = {key: default_collate(value) for key, value in batch.items()}

    
    return batch
