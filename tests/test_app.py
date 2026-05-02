import pytest
from unittest.mock import MagicMock
from openai import OpenAI

import torch
from PIL import Image
import numpy as np

def make_mock_llm_clinet(response_text:str):
    mock_choice = MagicMock()
    mock_choice.message.content= response_text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client

# ----------------------------------------------------
# Fixtures - of random image types and formats
# ----------------------------------------------------

@pytest.fixture
def rgb_image():
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image():
    """RGBA image with transparency channel — common for PNGs."""
    arr = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def grayscale_image():
    arr = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def small_image():
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def large_image():
    arr = np.random.randint(0, 255, (3024, 4032, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def wide_rectangle_image():
    arr = np.random.randint(0, 255, (480, 1280, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

# -----------------------------------------------------
# Tests
# -----------------------------------------------------

import sys
sys.path.insert(0,'src')
from app import preprocess_image, parse_plated, generate_response

ALL_FORMATS = [
    "rgb_image",
    "rgba_image", 
    "grayscale_image",
    "small_image",
    "large_image",
    "wide_rectangle_image",
]

@pytest.mark.parametrize("image_fixture", ALL_FORMATS)
def test_output_shape_always_224(image_fixture, request):
    """All image formats should produce (1, 3, 224, 224) tensor."""
    image  = request.getfixturevalue(image_fixture)
    tensor = preprocess_image(image)
    assert tensor.shape == (1, 3, 224, 224), f"{image_fixture} produced shape {tensor.shape}"

@pytest.mark.parametrize("image_fixture", ALL_FORMATS)
def test_pixel_values_normalized(image_fixture, request):
    """Output tensor values should always be in [0, 1]."""
    image  = request.getfixturevalue(image_fixture)
    tensor = preprocess_image(image)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

@pytest.mark.parametrize("image_fixture", ALL_FORMATS)
def test_output_is_float32(image_fixture, request):
    """Output tensor should always be float32 for model compatibility."""
    image  = request.getfixturevalue(image_fixture)
    tensor = preprocess_image(image)
    assert tensor.dtype == torch.float32

def test_parse_kcalpro():
    client = make_mock_llm_clinet("876,87")

    plated_amounts = parse_plated(
        "Patient was plated 87g protein and 876 kcal for lunch.",
        client
    )

    assert plated_amounts[0] == 876.0, f"Expected parser output kcal = 876.0 got {plated_amounts[0]}"
    assert plated_amounts[1] == 87.0, f"Expected parser output protein = 87.0 got {plated_amounts[1]}"

def test_parse_returns_zero_for_missing_values():
    client = make_mock_llm_clinet('540, null')
    plated_amounts = parse_plated(
        "blah, blah, blah",
        client
    )

    assert plated_amounts[0] == 540.0, f"Expected parser output kcal = 540.0 got {plated_amounts[0]}"
    assert plated_amounts[1] == 0, f"Expected parser output protein = 0 got {plated_amounts[1]}"

    