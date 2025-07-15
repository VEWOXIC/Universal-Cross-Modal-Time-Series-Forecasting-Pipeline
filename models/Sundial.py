import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    """
    Sundial Pre-trained Model Wrapper

    This class encapsulates the Sundial model from Hugging Face to make it 
    compatible with the existing evaluation framework. It handles the probabilistic
    forecasting output by calculating the median of the generated samples.
    """
    def __init__(self, configs):
        """
        Initializes the Sundial model wrapper.
        
        Args:
            configs (dotdict): Model configuration object. Must contain:
                               - pred_len (int): The prediction length.
                               - num_samples (int, optional): Number of samples to generate. Defaults to 20.
        """
        super(Model, self).__init__()
        
        # Get prediction length and number of samples from config
        self.model_name = configs.model_name
        self.pred_len = configs.pred_len
        self.num_samples = configs.num_samples

        # --- Load the pre-trained Sundial model ---
        self.device = f"cuda:{int(configs.gpu)}" if configs.gpu is not None else "cpu"
        
        # Note: Sundial also requires `trust_remote_code=True`
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
        )
        print(f"[ info ] {self.model_name} loaded successfully on device {next(self.model.parameters()).device}.")
        
        # Set to evaluation mode for inference
        self.model.eval()

    def forward(self, x):
        """
        Performs a forward pass (prediction) following Sundial's logic.

        Args:
            x (torch.Tensor): 
                Input context tensor. For univariate tasks, its shape is 
                [Batch, Input length] from the DataLoader.
            **kwargs: 
                Catches any extra arguments and ignores them.
        
        Returns:
            torch.Tensor: 
                Prediction tensor, reshaped to [Batch, Output length, 1] 
                to match the framework's expectation.
        """
        # Sundial, unlike Time-MoE, does not require explicit external normalization.
        # It takes the raw (or globally scaled) sequence directly.
        context = x

        # --- Model Generation ---
        # Generate multiple future sample trajectories.
        with torch.no_grad():
            output = self.model.generate(
                context,
                max_new_tokens=self.pred_len,
                num_samples=self.num_samples,
            )
        
        # The output shape is [Batch * num_samples, Input length + Output length]
        
        # --- Process Probabilistic Forecast ---
        
        # 1. Reshape the output to separate batch and sample dimensions.
        #    Shape -> [Batch, num_samples, Input length + Output length]
        batch_size = context.shape[0]
        output = output.reshape(batch_size, self.num_samples, -1)
        
        # 2. Extract only the newly generated (predicted) part of the sequences.
        #    Shape -> [Batch, num_samples, pred_len]
        sample_predictions = output[:, :, -self.pred_len:]

        # 3. To get a single point forecast, calculate the median across the samples.
        #    This is a common way to get a robust point estimate from a probabilistic forecast.
        #    Shape -> [Batch, pred_len]
        median_prediction = torch.quantile(sample_predictions, 0.5, dim=1)
        
        return median_prediction