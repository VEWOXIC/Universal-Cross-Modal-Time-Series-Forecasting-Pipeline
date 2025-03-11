import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from .spliter import timestamp_spliter, ratio_spliter
import multiprocessing as mp
from time import time

warnings.filterwarnings('ignore')

class Universal_Dataset(Dataset):
    def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
                 target='OT', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init
        self.spliter = spliter
        self.set_type = flag

        self.target = target
        self.scale = scale

        self.timestamp_col = timestamp_col

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.collect_all_data()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_path.endswith('.csv'):
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        elif self.data_path.endswith('.parquet'):
            df_raw = pd.read_parquet(os.path.join(self.root_path,
                                            self.data_path))
        else:
            raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')

        # convert the timestamp to datetime
        df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col])

        # apply the spliter
        train_data, val_data, test_data = self.spliter(df=df_raw)

        if self.set_type == 'train':
            self.data = train_data
        elif self.set_type == 'val':
            self.data = val_data
        elif self.set_type == 'test':
            self.data = test_data

        self.timestamp = self.data[self.timestamp_col].values
        self.data = self.data[self.target].values

        if self.scale:
            self.scaler.fit(train_data[self.target].values)
            self.data = self.scaler.transform(self.data)


    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []

        def process_data(i):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            return (self.data[s_begin:s_end], self.data[r_begin:r_end], 
            self.timestamp[s_begin:s_end], self.timestamp[r_begin:r_end])
        
        for i in range(len(self.data) - self.seq_len - self.pred_len + 1):
            x_data, y_data, x_time, y_time = process_data(i)
            self.x_data.append(x_data)
            self.y_data.append(y_data)
            self.x_time.append(x_time)
            self.y_time.append(y_time)

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]

        return seq_x, seq_y#, self.x_time[index], self.y_time[index]

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)