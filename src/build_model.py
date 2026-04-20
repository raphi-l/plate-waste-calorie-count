import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

import yaml

with open('configs/dish_data_config.yaml','r') as f:
    configs = yaml.safe_load(f)

TARGETS_NO = len(configs['dish_cols']) - 1  # -1 for dish_id

# -------------------------------------------
# Linear Regression Model - Sanity/Baseline
# -------------------------------------------

class LinearRegressor(nn.Module):
    def __init__(self, input_size: int = 224 * 224 * 3):
        super().__init__()
        self.flatten = nn.Flatten
        self.linear = nn.Linear(input_size, TARGETS_NO)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(x))


    
        
    







def build_model(model_config: dict) -> nn.module:
    model_type = model_crg['model']

    if model_type == 'linear':
        return LinearRegressor()
    
    elif model_type == 'efficientnet_b0':
        return torchvision.models.efficientnet_b0(
            weights = configs[model_type][weights]
        )
    
    elif model_type == 'efficientnet_b1':
        return torchvision.models.efficientnet_b1(
            weights = configs[model_type][weights]
        )

    

