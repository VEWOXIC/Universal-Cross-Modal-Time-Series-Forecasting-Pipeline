import pandas as pd
import os

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
