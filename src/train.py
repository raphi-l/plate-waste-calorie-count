import yaml
import argparse
import random
import numpy as np
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import mlflow
import mlflow.pytorch

from src.dataset import Nutrition5kDataset
from src.build_model import build_model
from src.evaluate import compute_metrics

# --------------------------------------------------------
# Configure
# --------------------------------------------------------

DATA_CONFIG_PATH = Path("configs/dish_data_config.yaml")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_configs():
    with open(DATA_CONFIG_PATH) as f:
        data_cfg = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH) as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg

# --------------------------------------------------------
# Connect to my Colab GPU
# --------------------------------------------------------
CHECKPOINT = "/content/drive/MyDrive/plate-intake/models/efficientnet_best.pt"

# resume from checkpoint if it exists
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT)
    build_model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 1

# --------------------------------------------------------
# Set up seeds
# --------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------
# Get data and loaders
# --------------------------------------------------------

def get_dataloaders(data_cfg: dict, model_cfg: dict):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    target_cols = data_cfg["targets"]   # ["total_calories", "total_protein"]

    train_ds = Nutrition5kDataset(
        metadata_paths=data_cfg["metadata_paths"],
        imagery_dir=data_cfg["imagery_dir"],
        split_file=data_cfg["train_split"],
        transform=train_transform,
    )
    test_ds = Nutrition5kDataset(
        metadata_paths=data_cfg["metadata_paths"],
        imagery_dir=data_cfg["imagery_dir"],
        split_file=data_cfg["test_split"],
        transform=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=model_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, test_loader, target_cols

# ---------------------------------------------------------------
# Multi-target Loss
# ---------------------------------------------------------------

def multi_target_loss(
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: list[float] = None, # weight importances of [kcal, protein]
        delta: float = 50.0, # error threshold for linear
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Per-target Huber loss (quadratic v linear for error below delta treshold)

    Returns:
        total: scalar loss for back propagation
        per_target: (2, ) tensor for each nutrient

    """
    huber = nn.HuberLoss(reduction='none', delta=delta)

    # shape: (BATCH, 2) -> rows = dishes, cols = [kcal_loss, pro_loss]
    losses_per_dish = huber(preds, targets)

    # shape: (2, ) -> [mean_kcal_loss, mean_pro_loss] over all batches
    loss_per_target = losses_per_dish.mean(dim=0)

    if weights is not None:
        w = torch.tensor(weights,
                         device=preds.device, dtype=torch.float32)
        total = (loss_per_target * w).sum()
    
    else:
        total = loss_per_target.mean()

    return total, loss_per_target

# ---------------------------------------------------------------
# Train One Epoch
# ---------------------------------------------------------------

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_weights: list[float],
        huber_delta: float,            # passed in from training config
        ) -> tuple[float, np.ndarray]:
 
    model.train()
    total_loss = 0.0
    total_loss_per_target = np.zeros(2)
 
    for batch in loader:
        images  = batch['image'].to(device)
        targets = batch['target'].to(device)
 
        optimizer.zero_grad()
 
        preds = model(images)
 
        loss, loss_per_target = multi_target_loss(
            preds, targets, loss_weights, delta=huber_delta
        )
 
        loss.backward()
        optimizer.step()
 
        n = len(images)  # number of dishes in this batch (smaller for last batch)
        total_loss            += loss.item() * n
        total_loss_per_target += loss_per_target.detach().cpu().numpy() * n
 
    n_total = len(loader.dataset)
 
    return (total_loss / n_total), (total_loss_per_target / n_total)

# ---------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------

def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        loss_weights: list[float],
        huber_delta: float,
        target_cols: list[str],
) -> tuple[float, dict]:
 
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
 
    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)
            targets = batch["target"].to(device)
 
            preds = model(images)
            loss, _ = multi_target_loss(
                preds, targets, loss_weights, delta=huber_delta
            )
 
            total_loss += loss.item() * len(images)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
 
    all_preds   = np.vstack(all_preds)     # (N, 2)
    all_targets = np.vstack(all_targets)   # (N, 2)
 
    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_targets, all_preds, target_cols)
 
    return avg_loss, metrics
 
# ---------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------
 
def run_experiment(
        config_name: str,
        data_cfg: dict,
        model_cfg: dict,
        target_cols: list[str],
):
    training_cfg = model_cfg["training"]
    exp_cfg      = model_cfg["configs"][config_name]
 
    set_seed(training_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print(f"\n{'='*60}")
    print(f"  Configuration : {config_name}")
    print(f"  Targets       : {target_cols}")
    print(f"  Device        : {device}")
    print(f"{'='*60}")
 
    train_loader, test_loader, _ = get_dataloaders(data_cfg, model_cfg)
 
    model = build_model(exp_cfg).to(device)
 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=exp_cfg["lr"],
    )
 
    loss_weights = training_cfg.get("loss_weights", None)
    huber_delta  = training_cfg.get("huber_delta", 50.0)   # read from config, default 50
 
    with mlflow.start_run(run_name=config_name):
 
        # log params
        mlflow.log_param("config_name",  config_name)
        mlflow.log_param("model_type",   exp_cfg["model"])
        mlflow.log_param("targets",      str(target_cols))
        mlflow.log_param("lr",           exp_cfg["lr"])
        mlflow.log_param("batch_size",   training_cfg["batch_size"])
        mlflow.log_param("epochs",       training_cfg["epochs"])
        mlflow.log_param("seed",         training_cfg["seed"])
        mlflow.log_param("loss_weights", str(loss_weights))
        mlflow.log_param("huber_delta",  huber_delta)
        mlflow.log_param("train_size",   len(train_loader.dataset))
        mlflow.log_param("test_size",    len(test_loader.dataset))
 
        for k, v in exp_cfg.items():
            if k not in ("model", "lr"):
                mlflow.log_param(k, v)
 
        best_val_loss = float("inf")
        Path("models").mkdir(exist_ok=True)
 
        for epoch in range(1, training_cfg["epochs"] + 1):
 
            train_loss, train_loss_per_target = train_one_epoch(
                model, train_loader, optimizer, device, loss_weights, huber_delta
            )
            val_loss, metrics = evaluate(
                model, test_loader, device, loss_weights, huber_delta, target_cols
            )
 
            # log per-epoch metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
 
            for i, col in enumerate(target_cols):
                short = col.replace("total_", "")
                mlflow.log_metric(f"train_mae_{short}", train_loss_per_target[i], step=epoch)
 
            for key, val in metrics.items():
                mlflow.log_metric(key, val, step=epoch)
 
            kcal_mae  = metrics.get("mae_calories",  0)
            prot_mae  = metrics.get("mae_protein",   0)
            kcal_mape = metrics.get("mape_calories", 0)
            prot_mape = metrics.get("mape_protein",  0)
 
            print(
                f"  Epoch {epoch:02d}/{training_cfg['epochs']} | "
                f"train={train_loss:.2f} | val={val_loss:.2f} | "
                f"kcal MAE={kcal_mae:.1f} ({kcal_mape:.1f}%) | "
                f"protein MAE={prot_mae:.1f}g ({prot_mape:.1f}%)"
            )
 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"models/{config_name}_best.pt"
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "val_loss":     val_loss,
                    "metrics":      metrics,
                    "target_cols":  target_cols,
                    "config":       exp_cfg,
                    "huber_delta":  huber_delta,
                }, ckpt_path)
 
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(ckpt_path)
        mlflow.pytorch.log_model(model, artifact_path=config_name)
 
        print(f"\n  Best val loss : {best_val_loss:.3f}")
        print(f"  Checkpoint    : {ckpt_path}")
 
    return best_val_loss
 
# ---------------------------------------------------------------
# Summarize all runs
# ---------------------------------------------------------------
 
def summarize_experiments():
    print(f"\n{'='*60}")
    print("  Experiment summary")
    print(f"{'='*60}")
 
    runs = mlflow.search_runs(
        experiment_names=["plate-intake-estimator"],
        order_by=["metrics.best_val_loss ASC"],
    )
 
    cols = [
        "tags.mlflow.runName",
        "metrics.best_val_loss",
        "metrics.mae_calories",
        "metrics.mae_protein",
        "params.model_type",
        "params.huber_delta",
    ]
    cols = [c for c in cols if c in runs.columns]
    print(runs[cols].to_string(index=False))
 
    best = runs.iloc[0]
    print(f"\n  Best run : {best['tags.mlflow.runName']}")
    print(f"  Val loss : {best['metrics.best_val_loss']:.3f}")
 
# ---------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/dish_data_config.yaml")
    parser.add_argument(
        "--run", default="all",
        help="linear_baseline | efficientnet_frozen | efficientnet_finetune | all"
    )
    args = parser.parse_args()
 
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
 
    target_cols = data_cfg["targets"]
 
    mlflow.set_experiment("plate-intake-estimator")
 
    configs_to_run = (
        list(model_cfg["configs"].keys())
        if args.run == "all"
        else [args.run]
    )
 
    results = {}
    for config_name in configs_to_run:
        best = run_experiment(config_name, data_cfg, model_cfg, target_cols)
        results[config_name] = best
 
    summarize_experiments()
 
if __name__ == "__main__":
    main()