import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from layers.RevIN import RevIN

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_gpt = configs.is_gpt
        self.RevIN = configs.RevIN
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.prompt_layer = nn.Linear(configs.d_model, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * (self.patch_num), configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer, self.prompt_layer):
            layer.to(device=f"cuda:{configs.gpu}") if configs.gpu is not None else layer.to(device='cpu')
            layer.train()
        
        self.rev_in = RevIN(num_features=1) # channel independent

    def move_to_device(self, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Custom move data to device
        """
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_hetero = batch_x_hetero.float().to(device)
        return batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

    def get_emb(self, x, tokens=None):
        if tokens is None:
            x = self.gpt2(inputs_embeds=x).last_hidden_state
            return x
        else:
            [a, b, c] = x.shape
            prompt_x = self.relu(self.prompt_layer(tokens))
            x_all = torch.cat((prompt_x, x), dim=1)
            x = self.gpt2(inputs_embeds=x_all).last_hidden_state
            return x[: , -b:, :]

    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # b, 1, seq_len
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #b, 1, patch_num, patch_size
        x = rearrange(x, 'b m n p -> (b m) n p') # b, patch_num, patch_size

        return x

    def forward(self, x, historical_events, **kwargs):
        historical_events = rearrange(historical_events, 'b l m -> b m l') # [b, 768, 15]
        historical_events = self.padding_patch_layer(historical_events) # [b, 768, 19] 
        historical_events = historical_events.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [b, 768, 3, 8]
        historical_events = historical_events.mean(dim=-1).squeeze() # [b, 768, 3]
        if len(historical_events.shape) == 2:
            historical_events = historical_events.unsqueeze(0)  # add [b] if batch size = 1
        historical_events = rearrange(historical_events, 'b l m -> b m l') # [b, 3, 768]

        B, L, M = x.shape # 4, 512, 1

        # handling multivariate
        if M > 1:
            historical_events = historical_events.repeat_interleave(M, dim=0)  # [B*M, 3, d_model]

        if self.RevIN:
            x = self.rev_in(x, 'norm').to(f"cuda:{configs.gpu}") if configs.gpu is not None else self.rev_in(x, 'norm').to('cpu')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
       
        x = self.get_patch(x)
        x = self.in_layer(x)

        outputs = self.get_emb(x, historical_events)
        outputs = self.out_layer(outputs.reshape(B*M, -1)) 
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        if self.RevIN:
            outputs = self.rev_in(outputs, 'denorm').to(f"cuda:{configs.gpu}") if configs.gpu is not None else self.rev_in(outputs, 'denorm').to('cpu')
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs
