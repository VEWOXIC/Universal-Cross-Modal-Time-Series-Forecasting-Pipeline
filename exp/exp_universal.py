from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate

import numpy as np
import torch
import torch.nn as nn

import os
import time
import warnings

import json

from tqdm import tqdm

warnings.filterwarnings('ignore')

class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)

    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Get the data for training, validation, or testing.
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type='loader')
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type='loader')
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type='loader')

        return data_loader

    def _forward_step(self, iter):
        """
        Forward step for the model.
        """
        # iteration: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
        batch_x = iter[0].float().to(self.device)
        batch_y = iter[1].float().to(self.device)
        timestamp_x = iter[2]
        timestamp_y = iter[3]
        batch_x_hetero = iter[4]
        batch_y_hetero = iter[5]
        hetero_x_time = iter[6]
        hetero_y_time = iter[7]
        hetero_general = iter[8]
        hetero_channel = iter[9]

        output = self.model(x=batch_x, historical_events =batch_x_hetero, news = batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)

        output = output[:, -self.args.output_len:, :]
        gt = batch_y

        return output, gt


    def train(self, setting):
        """
        Train the model.
        """
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')
        print(self.model)
        # self._get_profile(self.model)
        # print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/' + 'args.json', 'w') as f:
                        json.dump(self.args.__dict__, f)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # self.model.eval()

        # vali_loss = self.vali(test_data, test_loader, criterion, valinum='full')
        # print("Initial test loss: ", vali_loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.args.train_epochs}", unit='batch') as pbar:
                for i, iter in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                        
                    output, gt = self._forward_step(iter)

                    loss = criterion(output, gt)

                    loss.backward()

                    model_optim.step()

                    train_loss.append(loss.item())

                    if iter_count % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        pbar.set_postfix({'loss': f'{loss.item():.7f}', 'speed': f'{speed:.4f}s/iter', 'left time': f'{left_time:.4f}s'})
                        pbar.update(100)
                        iter_count = 0
                        time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.test(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, loader, criterion):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, iter in tqdm(enumerate(loader), total=len(loader), desc=f"Validating..."):
                    
                    output, gt = self._forward_step(iter)

                    loss = criterion(output, gt)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test(self, loaders, criterion, valinum='full'):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()
        for info, loader in loaders.items():
            info_loss=[]
            with torch.inference_mode():
                with torch.no_grad():
                    for i, iter in tqdm(enumerate(loader), total=len(loader), desc=f"Testing {info}"):
                        
                        output, gt = self._forward_step(iter)

                        loss = criterion(output, gt)
                        total_loss.append(loss.item())
                        info_loss.append(loss.item())
                        ######
                        if valinum == 'full':
                            pass
                        else:
                            if i == valinum:
                                break # only sample 2 batches for faster validation
                        ######
            print(f"Test loss for {info}: {np.average(info_loss)}")
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss