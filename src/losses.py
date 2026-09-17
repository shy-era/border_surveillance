import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary change detection to mitigate class imbalance.
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss.
    Loss = alpha * BCE + (1 - alpha) * Dice
    """
    def __init__(self, alpha: float = 0.5, pos_weight: float = 1.0, smooth: float = 1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        if self.pos_weight is not None:
            bce_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(device))
        else:
            bce_fn = nn.BCEWithLogitsLoss()

        bce_loss = bce_fn(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for focusing on hard-to-detect border structure changes.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = focal_weight * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()
