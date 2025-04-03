import importlib
import yaml
from utils.tools import dotdict
from .LLM_socket import LLM_Socket

def model_init(model_name, configs, all_args, is_LLM=False):

    if is_LLM:
        return LLM_Socket(configs)
    else:
        configs['seq_len']=all_args.input_len
        configs['pred_len'] = all_args.output_len

        data_configs = all_args.data_config
        try:
            configs['input_channel'] = data_configs.input_channel
        except: pass
        try:
            configs['base_T'] = data_configs.base_T
        except: pass
        try:
            configs['sampling_rate'] = data_configs.sampling_rate
        except: pass

        
        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'Model')
        return model_class(configs)
