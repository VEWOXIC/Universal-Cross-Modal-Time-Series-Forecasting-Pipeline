from data_provider.data_factory import data_provider
from data_provider.data_helper import data_buffer
from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt
from thop import profile
import tqdm, json
from torch.fft import rfft, irfft

from copy import deepcopy

warnings.filterwarnings('ignore')

class Exp_uni(Exp_Basic):
    def __init__(self, args):
        super(Exp_uni, self).__init__(args)

    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Get the data for training, validation, or testing.
        """
        data_set, data_loader = data_provider(self.args, flag, buffer=self.data_buffer)
        return data_set, data_loader

    def _get_profile(self, model):
        """
        Get the model profile including FLOPs and parameters.
        """
        _input = torch.randn(self.args.batch_size, self.args.seq_len, 1).to(self.device)
        macs, params = profile(model, inputs=(_input).to(self.device))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def train(self, setting):
        """
        Train the model.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
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

            for i, iter in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                    
                batch_x = iter[0].float().to(self.device)
                batch_y = iter[1].float().to(self.device)

                output = self.model(batch_x)

                loss = criterion(output[:, -self.args.output_len:, :], batch_y)

                loss.backward()

                model_optim.step()

                # loss = criterion(torch.cat(output, dim=-1), torch.cat(supervision,dim=-1)) 

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # save the self.args as json



            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.test(test_data, test_loader, criterion)

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

    def vali(self, data, loader, criterion):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, iter in tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"Validating..."):
                    
                    batch_x = iter[0].float().to(self.device)
                    batch_y = iter[1].float().to(self.device)

                    output = self.model(batch_x)

                    loss = criterion(batch_y, output[:, -self.args.output_len:, :])
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test(self, data, loaders, criterion, valinum='full'):
        """
        Validate the model on the validation dataset.
        """
        total_loss = []
        self.model.eval()
        for info, loader in loaders.items():
            info_loss=[]
            with torch.inference_mode():
                with torch.no_grad():
                    for i, iter in tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"Validating {info}"):
                        
                        batch_x = iter[0].float().to(self.device)
                        batch_y = iter[1].float().to(self.device)

                        output = self.model(batch_x)
                        # pred = output.detach().cpu().permute(0, 2, 1)
                        # true = batch_y.detach().cpu()

                        loss = criterion(batch_y, output[:, -self.args.output_len:, :])
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