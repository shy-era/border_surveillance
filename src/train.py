import os
import argparse
import time
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.model import create_model
from src.dataset import SatelliteChangeDataset
from src.losses import BCEDiceLoss
from src.metrics import calculate_metrics


def train_model(
    data_dir: str = 'data/samples',
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    backbone: str = 'resnet18',
    save_dir: str = 'checkpoints',
    val_ratio: float = 0.2
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Training on Device: {device} | Backbone: {backbone} ===")

    # Prepare Dataset
    full_dataset = SatelliteChangeDataset(root_dir=data_dir, image_size=256, is_train=True)
    if len(full_dataset) == 0:
        raise ValueError(f"No image pairs found in {data_dir}. Run `python -m src.generate_samples` first.")

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    if val_size > 0:
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    else:
        train_ds, val_ds = full_dataset, full_dataset

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset summary: Total={len(full_dataset)} | Train={len(train_ds)} | Val={len(val_ds)}")

    # Model, Optimizer, Loss, Scheduler
    model = create_model(backbone=backbone, pretrained=True).to(device)
    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for t1, t2, targets, _ in pbar:
            t1 = t1.to(device)
            t2 = t2.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation Phase
        model.eval()
        val_metrics = {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for t1, t2, targets, _ in val_loader:
                t1 = t1.to(device)
                t2 = t2.to(device)
                targets = targets.to(device)

                logits = model(t1, t2)
                batch_m = calculate_metrics(logits, targets)
                for k in val_metrics:
                    val_metrics[k] += batch_m[k]

        for k in val_metrics:
            val_metrics[k] /= max(1, num_val_batches)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | Val Prec: {val_metrics['precision']:.4f}"
        )

        # Save Checkpoints
        latest_path = os.path.join(save_dir, 'latest_model.pth')
        torch.save(model.state_dict(), latest_path)

        if val_metrics['f1'] >= best_f1:
            best_f1 = val_metrics['f1']
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f" -> Best checkpoint saved with F1={best_f1:.4f} at {best_path}")

    print("\n=== Training Completed Successfully! ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Siamese U-Net for Border Surveillance Change Detection")
    parser.add_argument('--data_dir', type=str, default='data/samples', help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='Backbone')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model weights')

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        save_dir=args.save_dir
    )
