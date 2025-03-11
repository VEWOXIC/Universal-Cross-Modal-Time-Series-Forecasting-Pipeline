import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Universal_Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, split=(7,1,2), timestamp_col='date',
                 target='OT', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init

        self.set_type = flag
        # check if split is string it should be in format "x:y:z"
        if isinstance(split, str):
            assert len(split.split(':')) == 3, "Split should be in format 'x:y:z'"
            split = [int(x) for x in split.split(':')]
        elif isinstance(split, tuple) or isinstance(split, list):
            assert len(split) == 3, "Split should be in format 'x:y:z', or list with length of 3"
        else:
            raise ValueError("Split should be in format 'x:y:z', or list with length of 3")

        # split mark
        self.train_split = split[0] / sum(split)
        self.val_split = split[1] / sum(split) + self.train_split    # [----train----] train_split [----val----] val_split [----test----]

        self.target = target
        self.scale = scale

        self.timestamp_col = timestamp_col

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.collect_all_data()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        raw_data_len = len(df_raw)

        train_split = int(raw_data_len * self.train_split)
        val_split = int(raw_data_len * self.val_split)

        if flag == 'train':
            border1 = 0
            border2 = train_split
        elif flag == 'val':
            border1 = train_split - self.seq_len
            border2 = val_split
        elif flag == 'test':
            border1 = val_split - self.seq_len
            border2 = -1

        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[0:train_split]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.timestamp_col]][border1:border2]
        df_stamp[self.timestamp_col] = pd.to_datetime(df_stamp[self.timestamp_col])

        self.data = data[border1:border2]
        self.data_stamp = data_stamp

    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []
        for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            self.x_data.append(self.data[s_begin:s_end])
            self.y_data.append(self.data[r_begin:r_end])
            self.x_time.append(self.data_stamp[s_begin:s_end]) 
            self.y_time.append(self.data_stamp[r_begin:r_end])

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]
        return seq_x, seq_y, self.x_time[index], self.y_time[index]

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)