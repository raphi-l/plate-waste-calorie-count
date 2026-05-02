import torch
import pytest
import yaml
from pathlib import Path

import sys
sys.path.insert(0,'src')
from build_model import build_model

# ---------------------------------------------
# Fixtures
# ---------------------------------------------

@pytest.fixture
def linear_model():
    return build_model({"model": 'linear_baseline'})

@pytest.fixture
def efficient_model():
    config_path = Path(__file__).parent.parent / 'models'/"best_model_report.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)['hyperparameters']

    return build_model(configs)

@pytest.fixture
def dummy_batch():
    return torch.rand(8, 3, 224, 224) #(B,C,H,W)

# ---------------------------------------------
# Tests
# ---------------------------------------------

def test_model_outputs(efficient_model, dummy_batch):
    efficient_model.eval()
    with torch.no_grad():
        preds = efficient_model(dummy_batch)
    assert preds.shape == (8,2), "Expected (8,2) but got {preds.shape}"

    assert preds.dtype == torch.float32, f"Expected output to be in format float32, got {preds.dtype}"

def test_model_output_plausible(efficient_model, dummy_batch):
    efficient_model.eval()
    with torch.no_grad():
        preds = efficient_model(dummy_batch)
    
    assert torch.isfinite(preds).all(), "Model output contains NaN or inf values"
    assert torch.all(preds > 0), "Model output contains negative values"