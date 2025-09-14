import re
from statistics import mode

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from utils.ChatTime_prompt import getPrompt
from utils.ChatTime_tools import Discretizer, Serializer


class Model(nn.Module):
    """
    ChatTime model for time series forecasting using large language models.
    
    This class implements a time series forecasting model that leverages pre-trained
    language models (specifically Llama) for prediction tasks. It converts time series
    data into text representations that can be processed by language models, then
    converts the text outputs back to numerical predictions.
    
    The model handles discretization of continuous time series values, serialization
    to text format, prompt engineering, and sampling from the language model to
    generate forecasts.
    
    Attributes:
        model_name (str): Name/path of the pre-trained language model
        task (str): Task type (e.g., 'long_term_forecast')
        device (str): Computing device for model execution
        hist_len (int): Length of historical context used for prediction
        pred_len (int): Length of prediction horizon
        max_pred_len (int): Maximum prediction length per iteration
        num_samples (int): Number of samples to generate from the model
        top_k (int): Top-k sampling parameter for text generation
        top_p (float): Top-p (nucleus) sampling parameter for text generation
        temperature (float): Temperature for sampling randomness control
        discretizer (Discretizer): Tool for converting continuous to discrete values
        serializer (Serializer): Tool for converting numerical data to text format
        model (LlamaForCausalLM): The pre-trained language model
        tokenizer (LlamaTokenizer): Tokenizer for the language model
    """
    def __init__(self, configs):
        """
        Initialize the ChatTime model with configuration parameters.
        
        Args:
            configs (object): Configuration object containing model parameters with attributes:
                - model_name (str): Path or name of the pre-trained language model
                - task (str): Task type for the model
                - gpu (int or None): GPU device number, None for CPU
                - hist_len (int): Historical sequence length
                - pred_len (int): Prediction sequence length
                - num_samples (int): Number of samples for text generation
                - top_k (int): Top-k parameter for sampling
                - top_p (float): Top-p parameter for nucleus sampling
                - temperature (float): Temperature for controlling randomness
        """

        super(Model, self).__init__()

        self.model_name = configs.model_name
        self.task = configs.task
        self.device = f"cuda:{int(configs.gpu)}" if configs.gpu is not None else "cpu"
        
        self.hist_len = configs.hist_len
        self.pred_len = configs.pred_len

        self.max_pred_len = 1000
        self.num_samples = configs.num_samples
        self.top_k = configs.top_k
        self.top_p = configs.top_p
        self.temperature = configs.temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        print(f"[ info ] {self.model_name} loaded successfully on device {next(self.model.model.parameters()).device}.")

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

    def move_to_device(self, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data tensors to the specified device (placeholder implementation).
        
        This method is intended to handle device placement for data tensors,
        but currently returns the inputs unchanged since ChatTime primarily
        processes data on the device where the language model is loaded.
        
        Args:
            batch_x (torch.Tensor): Input time series batch
            batch_y (torch.Tensor): Target time series batch  
            timestamp_x (object): Input timestamps
            timestamp_y (object): Target timestamps
            batch_x_hetero (object): Heterogeneous input data
            batch_y_hetero (object): Heterogeneous target data
            hetero_x_time (object): Heterogeneous input time data
            hetero_y_time (object): Heterogeneous target time data
            hetero_general (object): General heterogeneous data
            hetero_channel (object): Channel-specific heterogeneous data
            device (str): Target device for tensor placement
            
        Returns:
            tuple: All input arguments returned unchanged
        """
        return batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

    def forward(self, x, batch_y_hetero=None, hetero_general=None, hetero_channel=None, **kwargs):
        """
        Perform forward pass for time series forecasting using language model.
        
        This method implements the core forecasting logic by converting time series
        data to text, generating predictions via the language model, and converting
        results back to numerical format. It supports iterative prediction for
        long forecasting horizons and incorporates heterogeneous auxiliary data.
        
        Args:
            x (torch.Tensor): Input time series data with shape [batch_size, seq_len, n_features].
                            Contains the historical time series values to condition on.
            batch_y_hetero (list): Heterogeneous data corresponding to the prediction period.
                                 Contains auxiliary information like news, events, etc.
            hetero_general (list): General dataset-level heterogeneous information.
                                 Provides context about the overall dataset characteristics.
            hetero_channel (list): Channel-specific heterogeneous information.
                                 Contains metadata about individual time series channels.
        
        Returns:
            torch.Tensor or None: Predicted time series values with shape [batch_size, pred_len, n_features].
                                Returns None if prediction generation fails (empty predictions).
        
        Raises:
            ValueError: If hist_len or pred_len are not properly configured before calling forward.
        
        Process:
            1. Validates configuration parameters
            2. Combines heterogeneous context information
            3. Iteratively generates predictions up to pred_len:
               - Discretizes current series using Discretizer
               - Serializes to text format using Serializer
               - Creates structured prompt with context
               - Generates multiple samples from language model
               - Deserializes and post-processes predictions
               - Uses median aggregation across samples
            4. Concatenates all predictions and reshapes for output
        
        Note:
            The method handles variable-length predictions by iterating until the
            full prediction horizon is covered, with each iteration limited by
            max_pred_len to manage computational constraints.
        """
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        batch_size = x.shape[0]
        series = x
        prediction_list = []
        remaining = self.pred_len

        context = \
            hetero_general + hetero_channel + batch_y_hetero \
                if batch_y_hetero is not None and hetero_general is not None and hetero_channel is not None \
                    else None
            
        # print(context)

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                max_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
            )
            samples = pipe(serialized_series)

            pred_list = []
            for sample in samples:
                serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                
                if dispersed_prediction.size == 0:
                    print("Warning: Empty prediction array!")
                    return None
                
                pred = self.discretizer.inverse_discretize(dispersed_prediction)

                if len(pred) < min(remaining, self.max_pred_len):
                    pred = np.concatenate([pred, np.full(min(remaining, self.max_pred_len) - len(pred), np.NaN)])

                pred_list.append(pred[:min(remaining, self.max_pred_len)])

            prediction = np.nanmedian(pred_list, axis=0)
            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)  # [pred_len,]
        prediction = prediction.reshape(batch_size, -1)  # [batch_size, pred_len]

        # transform to tensor
        return torch.tensor(prediction, dtype=torch.float32)
    
    """
    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )
        samples = pipe(serialized_series)

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response
    """