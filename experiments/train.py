import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import random
import platform
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.dataset import get_dataloaders
from models.baseline_unet import BaselineUNet
from models.dilated_unet import DilatedUNet
from models.metrics import CombinedLoss, MetricTracker, count_parameters

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_environment_info():
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count()
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, metric_tracker, device):
    model.eval()
    running_loss = 0.0
    metric_tracker.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            metric_tracker.update(preds, masks)

    metrics = metric_tracker.get_metrics()
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics

def train_experiment(model_type="baseline", data_dir="d:/ACV/data/PennFudanPed",
                     output_dir="d:/ACV/results", epochs=5, batch_size=8,
                     lr=1e-3, seed=42, patience=3, img_size=(128, 128)):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(output_dir, model_type)
    os.makedirs(exp_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"Training Model: {model_type.upper()}", flush=True)
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch: {batch_size} | Res: {img_size}", flush=True)
    print("=" * 60, flush=True)

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size, seed=seed, img_size=img_size)

    if model_type == "baseline":
        model = BaselineUNet(in_channels=3, num_classes=2)
    elif model_type == "dilated":
        model = DilatedUNet(in_channels=3, num_classes=2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    param_info = count_parameters(model)

    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    metric_tracker = MetricTracker(num_classes=2)

    config = {
        "model_type": model_type,
        "seed": seed,
        "resolution": list(img_size),
        "batch_size": batch_size,
        "optimizer": "AdamW (lr=0.001, weight_decay=0.0001)",
        "scheduler": f"CosineAnnealingLR (T_max={epochs}, eta_min=1e-5)",
        "loss": "Combined CrossEntropy (0.5) + Dice Loss (0.5)",
        "augmentations": "RandomHorizontalFlip(p=0.5), RandomRotation(-10..+10 deg), ColorJitter(+-0.15)",
        "stopping_rule": f"Max epochs {epochs} with early stopping patience={patience}",
        "parameters": param_info,
        "environment": get_environment_info()
    }

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_miou = -1.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metric_tracker, device)
        scheduler.step()
        epoch_time = time.time() - t0

        val_loss = val_metrics["loss"]
        val_miou = val_metrics["miou"]
        val_bg_iou = val_metrics["per_class_iou"]["class_0"]
        val_ped_iou = val_metrics["per_class_iou"]["class_1"]
        val_acc = val_metrics["pixel_accuracy"]

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_miou": round(val_miou, 4),
            "val_bg_iou": round(val_bg_iou, 4),
            "val_ped_iou": round(val_ped_iou, 4),
            "val_pixel_acc": round(val_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time_sec": round(epoch_time, 2)
        }
        history.append(record)

        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "training_log.csv"), index=False)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) | "
              f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
              f"Val mIoU: {val_miou*100:.2f}% (Ped IoU: {val_ped_iou*100:.2f}%)", flush=True)

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    total_training_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))

    summary = {
        "best_val_miou": best_miou,
        "total_epochs": len(history),
        "total_time_sec": round(total_training_time, 2)
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Finished {model_type} in {total_training_time:.2f}s | Best Val mIoU: {best_miou*100:.2f}%\n", flush=True)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "dilated", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model in ["baseline", "both"]:
        train_experiment("baseline", epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    if args.model in ["dilated", "both"]:
        train_experiment("dilated", epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
