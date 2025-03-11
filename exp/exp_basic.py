import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from data_provider.data_helper import data_buffer
from thop import profile


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        if args.disable_buffer:
            self.data_buffer = None
        else:
            self.data_buffer = data_buffer()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    
    def _select_optimizer(self):
        """
        Select the optimizer for training.
        """
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        Select the loss function for training.
        """
        if self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'mse':
            return nn.MSELoss()
        
    
    def _get_profile(self, model):
        """
        Get the model profile including FLOPs and parameters.
        """
        _input = torch.randn(self.args.batch_size, self.args.seq_len, 1).to(self.device)
        macs, params = profile(model, inputs=(_input).to(self.device))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
