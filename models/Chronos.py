import torch
import torch.nn as nn

# pip install chronos-forecasting
try:
    from chronos import BaseChronosPipeline
except ImportError:
    raise ImportError(
        "Chronos model requires the 'chronos-forecasting' library. "
        "Please install it using 'pip install chronos-forecasting' or 'pip install chronos-forecasting[gpu]'"
    )

class Model(nn.Module):
    """
    Chronos for Time Series Forecasting.
    
    This class is a wrapper for the pre-trained Chronos models from Hugging Face,
    making it compatible with the existing TSF/TGTSF framework.
    """
    def __init__(self, configs):
        """
        Initializes the Chronos model wrapper.

        Args:
            configs (dotdict): A dictionary-like object containing model configurations.
                               It is expected to contain 'pred_len' and 'task'.
        """
        super(Model, self).__init__()
        
        # Store necessary configurations
        self.num_samples = configs.num_samples
        self.model_name = configs.model_name
        self.task_name = configs.task

        self.pred_len = configs.pred_len
        self.device = f"cuda:{int(configs.gpu)}" if configs.gpu is not None else "cpu"

        # Initialize the Chronos pipeline from a pre-trained model.
        # device_map="auto" will automatically place the model on the best available device (GPU or CPU).
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        )
        print(f"[ info ] {self.model_name} loaded successfully on device {next(self.pipeline.model.parameters()).device}.")
        
        # Set to evaluation mode for inference
        self.pipeline.model.eval()

    def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        # Chronos does not need to move data to device
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
    
    def forward(self, x):
        """
        Forward pass for the Chronos model. It takes the input time series
        and returns a forecast for the prediction length.

        Args:
            x (torch.Tensor): Input time series data with shape 
                              (batch_size, input_length) or (batch_size, input_length, 1).
            **kwargs: Catches other potential arguments from the data loader 
                      (e.g., 'news', 'channel_description' for TGTSF tasks), 
                      which are ignored by this model.

        Returns:
            torch.Tensor: The forecast with shape 
                          (batch_size, prediction_length, 1).
        """
        if self.task_name == 'TGTSF':
            print("[ Warning ] Chronos model does not support TGTSF tasks.")
            return
        
        # --- Data Shape Validation and Preparation (Corrected) ---
        # The pipeline expects a 2D context tensor: (batch_size, context_length).
        # The data loader might provide a 2D or 3D tensor.
        
        context = None
        if x.dim() == 3:
            # Input is (batch, seq_len, features). We expect features=1 for univariate series.
            if x.shape[2] != 1:
                raise ValueError(
                    f"Chronos model expects a univariate time series (features=1), "
                    f"but received 3D input with shape {x.shape}"
                )
            # Squeeze the last dimension to get (batch, seq_len) for the pipeline.
            context = x.squeeze(-1)
        elif x.dim() == 2:
            # Input is already (batch, seq_len), which is what the pipeline needs.
            context = x
        else:
            # Any other shape is unsupported.
            raise ValueError(
                f"Unsupported input shape for Chronos model: {x.shape}. "
                f"Expected a 2D (batch_size, seq_len) or 3D (batch_size, seq_len, 1) tensor."
            )
        
        # --- Prediction ---
        # The pipeline's predict method returns a tensor of shape:
        # (batch_size, num_samples, prediction_length)
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=self.pred_len
        )
        
        # --- Output Formatting ---
        # Take the median across the generated samples for a robust point forecast.
        # shape: [Batch, Pred_len]
        point_forecast = torch.median(forecast, dim=1).values
        
        return point_forecast