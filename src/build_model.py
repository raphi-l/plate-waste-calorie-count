import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

import yaml

_config_path = Path(__file__).parent.parent / 'configs' / 'dish_data_config.yaml'
with open(_config_path, 'r') as f:
    configs = yaml.safe_load(f)

if 'dish_cols' not in configs:
    raise KeyError("'dish_cols' key missing from dish_data_config.yaml")

TARGETS_NO = len(configs['dish_cols']) - 1  # -1 for dish_id

# -------------------------------------------
# Linear Regression Model - Sanity/Baseline
# -------------------------------------------

class LinearRegressor(nn.Module):
    """Base-line simple regression for sanity checking"""
    def __init__(self, input_size: int = 224 * 224 * 3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, TARGETS_NO)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(x))

class EfficientNetRegressor(nn.Module):
    def __init__(self,
                 layers_to_unfreeze: int = 0,
                 dropout: float = 0.4,
                 variant: int = 0,
                 final_feature_dim: int = 256):
        
        super().__init__()

        if not 0 <= variant <= 7:
            raise ValueError(f"variant must be 0–7, got {variant}")
        model_name = f'efficientnet_b{variant}'
        model_func = getattr(models, model_name)

        weight_name = f'EfficientNet_B{variant}_Weights'
        weights = getattr(models, weight_name).DEFAULT

        backbone = model_func(weights = weights)
        feature_dim = backbone.classifier[1].in_features

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # replace classifier head with regression
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, final_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(final_feature_dim, TARGETS_NO)
        )

        for param in self.features.parameters():
            param.requires_grad = False

        if layers_to_unfreeze > 0:
            layers = list(self.features.children())
            for layer in layers[-layers_to_unfreeze:]:
                is_batch_norm = isinstance(layer, (nn.BatchNorm1d,
                                                   nn.BatchNorm2d,
                                                   nn.BatchNorm3d))
                if not is_batch_norm:
                    for param in layer.parameters():
                        param.requires_grad = True
            print(f"[INFO] Unfreezing top {layers_to_unfreeze} of model.")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.head(x)
        

def build_model(model_cfg: dict) -> nn.Module:
    """
    Constructs a model from config.
    
    Args:
        model_cfg: Dict containing 'model', 'variant', 'weights', 'lr', 
            'optimizer', 'freeze_backbone', 'unfreeze_epoch', and 'dropout'.
            
    Returns:
        The initialized nn.Module.
    """
    
    model_type = model_cfg['model']

    if model_type == 'linear_baseline':
        return LinearRegressor()
    
    elif model_type == "efficientnet":
        return EfficientNetRegressor(
            layers_to_unfreeze=model_cfg.get("layers_to_unfreeze", 0),
            dropout=model_cfg.get("dropout", 0.4),
            variant=model_cfg.get("variant", 0),
        )
    
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: linear or efficientnet"
        )

