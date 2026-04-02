from timecopilot import TimeCopilot
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import torch
import torch.nn as nn
import pandas as pd


class Model(nn.Module):
    """
    TimeCopilot
    """
    def __init__(self, configs):
        """
        Initializes
        
        Args:
            configs (dotdict): Model configuration object, must contain 'pred_len'.
        """
        super(Model, self).__init__()
        
        # Get prediction length from config
        self.name = configs.name
        self.pred_len = configs.pred_len
        self.freq = 'H' if configs.freq == "1h" else "H"
        self.base_T = configs.base_T
        self.model = model = OpenAIChatModel(
            configs.base_model_name,
            provider=OpenAIProvider(
                base_url=configs.base_url,
                api_key=configs.api_key,
            ),
        )
        self.agent = TimeCopilot(llm=model, retries=3)

    def forward(self, x, timestamp_x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input time series data.
            timestamp_x (Tensor): Timestamps corresponding to batch_x.
        
        Returns:
            Tensor: Predicted future values of the time series.
        """
        # convert to df and send to agent
        # DataFrame must include at least the following columns:

        # unique_id: Unique identifier for each time series (string)
        # ds: Date column (datetime format)
        # y: Target variable for forecasting (float format)

        y_values = x.detach().cpu().numpy().flatten()
        ds_values = timestamp_x.detach().cpu().numpy().flatten()
        
        ds_str = ds_values.astype(str)
        ds_datetime = pd.to_datetime(ds_str, format='%Y%m%d%H%M%S')
        
        # 3. 构建 DataFrame
        df = pd.DataFrame({
            'unique_id': self.name,
            'ds': ds_datetime,
            'y': y_values
        })

        # print(df)

        result = self.agent.forecast(df=df, freq='H', h=self.pred_len, seasonality=self.base_T)
        result_df = result.fcst_df

        y_values = result_df.iloc[:, 2].values
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0)

        return y_tensor