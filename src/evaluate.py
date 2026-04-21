import numpy as np
from sklearn.metrics import r2_score

def compute_metrics(
        targets: np.ndarray,
        preds: np.ndarray, 
        target_cols: list[str],) -> dict:
    """
    Compute MAE, MAPE, and R^2 for each target.

    Returns a flat dict of metrics:
        mae_calories, mape_calories, r2_calories,
        mae_protein,  mape_protein,  r2_protein
    """
    metrics = {}

    for i, col in enumerate(target_cols):

        # shorten "total_calories" → "calories", "total_protein" → "protein"
        short = col.replace("total_", "")

        true_vals  = targets[:, i]   # (N,) ground truth for this target
        pred_vals  = preds[:, i]     # (N,) predictions for this target

        # average absolute error in original units (kcal or grams)
        mae = float(np.mean(np.abs(true_vals - pred_vals)))

        # small epsilon avoids division by zero for any zero-valued labels
        mape = float(
            np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100
        )

        r2 = float(r2_score(true_vals, pred_vals))

        metrics[f"mae_{short}"]  = mae
        metrics[f"mape_{short}"] = mape
        metrics[f"r2_{short}"]   = r2

    return metrics


def print_metrics(
        metrics: dict,
        target_cols: list[str]):
    
    """
    Print a metrics dict to console.
    Called at end of training to summarise best model performance.
    """

    print("\n" + "="*50)
    print("  Evaluation results")
    print("="*50)

    for col in target_cols:
        shorten_str = col.replace("total_", "")
        mae  = metrics.get(f"mae_{shorten_str}",  0)
        mape = metrics.get(f"mape_{shorten_str}", 0)
        r2   = metrics.get(f"r2_{shorten_str}",   0)

        print(f"\n  {shorten_str.capitalize()}")
        print(f"    MAE  : {mae:.2f}  ({'kcal' if shorten_str == 'calories' else 'g'})")
        print(f"    MAPE : {mape:.1f}%")
        print(f"    R^2   : {r2:.3f}")

    print("="*50 + "\n")


def load_and_evaluate(model, 
                      loader,
                      device,
                      target_cols: list[str],) -> dict:
    """
    Run inference on a DataLoader and return metrics.
    Useful for evaluating a loaded checkpoint outside of train.py.

    Example usage in a notebook:
        ckpt   = torch.load("models/efficientnet_best.pt")
        model.load_state_dict(ckpt["model_state"])
        metrics = load_and_evaluate(model, test_loader, device, target_cols)
    """
    import torch

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)
            targets = batch["target"].to(device)

            preds = model(images)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds   = np.vstack(all_preds)    # (N, 2)
    all_targets = np.vstack(all_targets)  # (N, 2)

    return compute_metrics(all_targets, all_preds, target_cols)