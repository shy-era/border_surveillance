import torch
import numpy as np
from typing import Dict, Tuple


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute change detection metrics from model predictions and ground truth.

    Args:
        preds: Tensor of logits or probabilities (B, 1, H, W) or (B, H, W)
        targets: Binary ground truth tensor (B, 1, H, W) or (B, H, W)
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary with IoU, F1, Precision, Recall, Accuracy
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    # Convert logits to binary mask if needed
    if preds.dtype == torch.float32:
        probs = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
        bin_preds = (probs >= threshold).long()
    else:
        bin_preds = (preds >= threshold).long()

    bin_targets = (targets >= 0.5).long()

    # Flatten
    y_pred = bin_preds.view(-1).cpu().numpy()
    y_true = bin_targets.view(-1).cpu().numpy()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    smooth = 1e-6

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }
