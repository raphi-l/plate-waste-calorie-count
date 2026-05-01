import pytest
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0,'src')

# ---------------------------------------------------
# Fixtures
# ---------------------------------------------------

@pytest.fixture
def tmp_image(tmp_path):
    img_path = tmp_path / "test_plate.jpg"

    # make random array shape (H,W,C) with into 0 to 255)
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return tmp_path, img_path

@pytest.fixture
def dataset_instance(tmp_path, tmp_image):
    """
    Create a Nutrition5kDataset instance for testing
    mocks the file system dependencies
    """
    sys.path.insert(0,'src')
    from dataset import Nutrition5kDataset

    tmp_dir, _ = tmp_image

    # create a fake dish folder with rbg.png
    dish_dir = tmp_path / "imagery" / "dish_999999"
    dish_dir.mkdir(parents=True)
    arr = np.random.randint(0,255,(480,640,3), dtype=np.uint8)
    Image.fromarray(arr).save(dish_dir / "rgb.png")

    # fake metadata csv
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "dish_000000,450.0,380.0,18.0,55.0,22.0,3\n"
        "dish_999999,820.0,650.0,34.0,90.0,40.0,5\n"
    )

    #fake split file
    split_path = tmp_path / "train.txt"
    split_path.write_text("dish_000000\ndish_999999\n")

    ds = Nutrition5kDataset(
        metadata_paths=[str(metadata_path)],
        imagery_dir=str(tmp_path / "imagery"),
        split_file = str(split_path)
    )
    return ds

# ------------------------------------------------
# TESTS
# ------------------------------------------------

def test_image_resizes_to_expected_shape(dataset_instance):
    """_load_image should return (224, 224, 3) float32 array."""
    dish_id = dataset_instance.metadata.iloc[0]["dish_id"]
    arr = dataset_instance._load_image(dish_id)
    assert arr.shape == (224, 224, 3)


def test_pixel_values_normalized_to_0_1(dataset_instance):
    """All pixel values should be float32 in [0, 1]."""
    dish_id = dataset_instance.metadata.iloc[0]["dish_id"]
    arr = dataset_instance._load_image(dish_id)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_missing_image_raises_error(dataset_instance):
    """Loading a nonexistent dish_id should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        dataset_instance._load_image("dish_nonexistent")


def test_load_metadata_does_not_mutate_csv(tmp_path):
    """
    _load_metadata reads and filters data but should never
    write back to or modify the original CSV file.
    """
    sys.path.insert(0,'src')
    from dataset import Nutrition5kDataset

    metadata_path = tmp_path / "metadata.csv"
    original_content = (
        "dish_001,450.0,380.0,18.0,55.0,22.0,3\n"
        "dish_002,820.0,650.0,34.0,90.0,40.0,5\n"
    )
    metadata_path.write_text(original_content)

    # instantiate a minimal dataset to get access to the method
    split_path = tmp_path / "train.txt"
    split_path.write_text("")

    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()

    # call _load_metadata directly
    ds = Nutrition5kDataset.__new__(Nutrition5kDataset)
    ds.imagery_dir  = imagery_dir
    ds.transform    = None
    ds.target_size  = (224, 224)

    ds._load_metadata(str(metadata_path))

    # confirm file unchanged
    assert metadata_path.read_text() == original_content






