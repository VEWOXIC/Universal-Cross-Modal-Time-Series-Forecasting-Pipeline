"""
Example script demonstrating how to add a new model to the time series forecasting framework.

This script:
1. Creates a simple new model 'SimpleLinear'
2. Creates a model configuration file
3. Shows how to run training with the new model
"""

import os
import torch
import torch.nn as nn
import argparse

# Step 1: Define the new model
# This would typically go in models/SimpleLinear.py
class Model(nn.Module):
    """
    A simple linear model for time series forecasting.
    This is just a demonstration - in practice, you would create this file in the models/ directory.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Extract configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Define a simple linear model
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        
        # Optional: add dropout for regularization
        self.dropout = nn.Dropout(configs.get('dropout', 0.1))
        
    def forward(self, x, **kwargs):
        """Forward pass for the simple linear model.
        
        Args:
            x: Input data [Batch, Input length, Channel]
            **kwargs: Additional arguments not used by this model but required for compatibility
            
        Returns:
            Output prediction [Batch, Output length, Channel]
        """
        # Reshape for linear layer: [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)
        
        # Apply linear transformation to each channel independently
        output = self.linear(x)
        
        # Apply dropout for regularization
        output = self.dropout(output)
        
        # Reshape back to [Batch, Output length, Channel]
        output = output.permute(0, 2, 1)
        
        return output

def create_model_config():
    """
    Creates a model configuration file for the SimpleLinear model.
    
    This would typically be saved as model_configs/general/SimpleLinear.yaml
    """
    config_dir = 'model_configs/general'
    os.makedirs(config_dir, exist_ok=True)
    
    # Create the configuration file
    config_path = os.path.join(config_dir, 'SimpleLinear.yaml')
    with open(config_path, 'w') as f:
        f.write("""# SimpleLinear model configuration
model: SimpleLinear
individual: False
enc_in: 1
dropout: 0.1
task: TSF
""")
    
    print(f"Created model configuration at: {config_path}")
    return config_path

def create_model_file():
    """
    Creates the model file in the models directory.
    """
    # Create the model file
    model_path = 'models/SimpleLinear.py'
    
    # Get the model class definition
    model_code = """import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    A simple linear model for time series forecasting.
    \"\"\"
    def __init__(self, configs):
        super(Model, self).__init__()
        # Extract configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Define a simple linear model
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        
        # Optional: add dropout for regularization
        self.dropout = nn.Dropout(configs.get('dropout', 0.1))
        
    def forward(self, x, **kwargs):
        \"\"\"Forward pass for the simple linear model.
        
        Args:
            x: Input data [Batch, Input length, Channel]
            **kwargs: Additional arguments not used by this model but required for compatibility
            
        Returns:
            Output prediction [Batch, Output length, Channel]
        \"\"\"
        # Reshape for linear layer: [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)
        
        # Apply linear transformation to each channel independently
        output = self.linear(x)
        
        # Apply dropout for regularization
        output = self.dropout(output)
        
        # Reshape back to [Batch, Output length, Channel]
        output = output.permute(0, 2, 1)
        
        return output
"""
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Write the model file
    with open(model_path, 'w') as f:
        f.write(model_code)
    
    print(f"Created model file at: {model_path}")
    return model_path

def main():
    """
    Main function demonstrating the complete workflow of adding a new model.
    """
    print("=== Example: Adding a New Model to the Framework ===")
    
    # Step 1: Create the model file
    model_path = create_model_file()
    
    # Step 2: Create the model configuration
    config_path = create_model_config()
    
    # Step 3: Show how to run training with the new model
    print("\n=== How to Train the New Model ===")
    
    # Using PyTorch pipeline
    print("\nTo train with PyTorch:")
    print(f"python run.py --model SimpleLinear --model_config {config_path} --data_config data_configs/fullsolar.yaml --input_len 96 --output_len 24")
    
    # Using PyTorch Lightning pipeline
    print("\nTo train with PyTorch Lightning:")
    print(f"python run_lightning.py --model SimpleLinear --model_config {config_path} --data_config data_configs/fullsolar.yaml --input_len 96 --output_len 24")
    
    # Using multi-GPU training
    print("\nTo train with multiple GPUs (Lightning):")
    print(f"python run_lightning.py --model SimpleLinear --model_config {config_path} --data_config data_configs/fullsolar.yaml --input_len 96 --output_len 24 --use_multi_gpu --devices 0,1,2,3")
    
    print("\n=== Model Structure ===")
    print("The SimpleLinear model consists of a single linear layer that projects from input_len to output_len.")
    print("This is a minimal example - in practice, you would likely want to create more sophisticated models.")
    
    print("\n=== Next Steps ===")
    print("1. Customize the model architecture in models/SimpleLinear.py")
    print("2. Adjust the configuration in model_configs/general/SimpleLinear.yaml")
    print("3. Train the model using one of the commands above")
    print("4. Evaluate the model's performance")

if __name__ == "__main__":
    main() 