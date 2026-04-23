import pandas as pd
import numpy as  np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import yaml

class Nutrition5kDataset(Dataset):
    """Creates PyTorch Dataset in (C, H, W) from overhead RGB images
    Regression target: total_calories, total_protein per dish
    """

    def __init__(self,
                 metadata_paths: list[str], #cafe1 or cafe2
                 imagery_dir: str,
                 split_file: str,
                 transform=None,
                 target_size: tuple = (224,224),
                 ):
        self.imagery_dir = Path(imagery_dir)
        self.transform = transform
        self.target_size = target_size

        with open(split_file) as f:
            # remove lead_trailing spaces if able
            valid_ids = set(line.strip() for line in f if line.strip())
        
        dfs = []
        for path in metadata_paths:
            df = self._load_metadata(path)
            dfs.append(df)
        metadata = pd.concat(dfs, ignore_index=True)

        metadata = metadata[metadata['dish_id'].isin(valid_ids)]
        metadata = metadata[
            metadata['dish_id'].apply(self._has_rgb_png)
            ].reset_index(drop=True)
        
        self.metadata = metadata
    
    def _load_metadata(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=0)

        _cfg_path = Path(__file__).parent.parent / 'configs' / 'dish_data_config.yaml'
        with open(_cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        dish_cols = config['dish_cols']

        df.columns = df.columns.str.strip()
        df = df[dish_cols].copy()

        df = df.dropna(subset=['total_calories', 'total_protein'])

        df = df[
            (df['total_calories'] > 0) &
            (df['total_calories'] < 2000) &
            (df['total_protein'] >= 0) &
            (df['total_protein'] < 60)
        ]

        return df.reset_index(drop=True)
    
    def _has_rgb_png(self, dish_id: str) -> bool:
        return (self.imagery_dir / dish_id / "rgb.png").exists()
    
    def _load_image(self, dish_id:str) -> np.ndarray:
        """return np.array image RGB with normalized pixle valuess"""
        path = self.imagery_dir / dish_id / 'rgb.png'
        img = Image.open(path).convert("RGB")
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        dish_id = row['dish_id']

        image = self._load_image(dish_id)
        image = torch.from_numpy(image).permute(2,0,1)

        if self.transform:
            image = self.transform(image)

        # kcal_target = torch.tensor(row['total_calories'])
        # pro_target = torch.tensor(row['total_protein'])

        target = torch.tensor(
            [row["total_calories"], row["total_protein"]],
            dtype=torch.float32
        )

        return {
            'image': image,
            'target': target,  # shape (2, )
            'dish_id': dish_id
        }
    




        