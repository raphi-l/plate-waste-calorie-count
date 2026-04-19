import pandas as pd
import numpy as  np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

class Nutrition5KDataset(Dataset):
    """Creates PyTorch Dataset in (C, H, W) from overhead RGB images
    Regression target: total_calories, total_protein per dish
    """

    def __init__(self,
                 metadata_paths: list[str],
                 imagery_dir: str,
                 split_file: str,
                 transform=None,
                 target_size: tuple = (224,224),
                 ):
        self.imagery_dir = Path(self.imagery_dir),
        self.transform = transform,
        self.target_size = target_size

        with open(split_file) as f:
            valid_ids = set(line.strip() for line in f if line.strip())
        
        dfs = []
        for path in metadata_paths:
            df = self._load_metadata(path)
            dfs.append(df)
        metadata = pd.concat(dfs, ignore_index=True)

        metadata = metadata[metadata['dish_id'].isin(valid_ids)]
        metadata = metadata[
            metadata['dish_id'].apply(self._has_image)
            ].reset_index(drop=True)
        
        self.metadata = metadata
    
    



        