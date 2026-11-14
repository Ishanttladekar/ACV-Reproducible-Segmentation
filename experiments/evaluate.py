import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import pandas as pd
import numpy as np

from dataset.dataset import get_dataloaders
from models.baseline_unet import BaselineUNet
from models.dilated_unet import DilatedUNet
from models.metrics import MetricTracker, count_parameters

def evaluate_model(model, loader, device):
    model.eval()
    metric_tracker = MetricTracker(num_classes=2)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            metric_tracker.update(preds, masks)

    return metric_tracker.get_metrics()

def run_evaluation(data_dir="d:/ACV/data/PennFudanPed",
                   results_dir="d:/ACV/results",
                   seed=42,
                   img_size=(128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print(f"Test Set Evaluation on {device}", flush=True)
    print("=" * 60, flush=True)

    _, _, test_loader = get_dataloaders(data_dir, batch_size=1, seed=seed, img_size=img_size)

    baseline_model = BaselineUNet(in_channels=3, num_classes=2)
    baseline_weights = os.path.join(results_dir, "baseline", "best_model.pth")
    baseline_model.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline_model.to(device)
    base_params = count_parameters(baseline_model)["total_parameters"]
    base_metrics = evaluate_model(baseline_model, test_loader, device)

    dilated_model = DilatedUNet(in_channels=3, num_classes=2)
    dilated_weights = os.path.join(results_dir, "dilated", "best_model.pth")
    dilated_model.load_state_dict(torch.load(dilated_weights, map_location=device))
    dilated_model.to(device)
    dil_params = count_parameters(dilated_model)["total_parameters"]
    dil_metrics = evaluate_model(dilated_model, test_loader, device)

    rows = [
        {
            "Architecture": "Baseline U-Net (d=1)",
            "Parameters": base_params,
            "Background IoU (%)": round(base_metrics["per_class_iou"]["class_0"] * 100, 2),
            "Pedestrian IoU (%)": round(base_metrics["per_class_iou"]["class_1"] * 100, 2),
            "mIoU (%)": round(base_metrics["miou"] * 100, 2),
            "Mean Dice (%)": round(base_metrics["mean_dice"] * 100, 2),
            "Pixel Accuracy (%)": round(base_metrics["pixel_accuracy"] * 100, 2),
        },
        {
            "Architecture": "Dilated U-Net (d=2, 4)",
            "Parameters": dil_params,
            "Background IoU (%)": round(dil_metrics["per_class_iou"]["class_0"] * 100, 2),
            "Pedestrian IoU (%)": round(dil_metrics["per_class_iou"]["class_1"] * 100, 2),
            "mIoU (%)": round(dil_metrics["miou"] * 100, 2),
            "Mean Dice (%)": round(dil_metrics["mean_dice"] * 100, 2),
            "Pixel Accuracy (%)": round(dil_metrics["pixel_accuracy"] * 100, 2),
        }
    ]

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "comparative_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("\nTest Set Results:", flush=True)
    print(df.to_string(index=False), flush=True)

    full_results = {
        "baseline": {
            "parameters": base_params,
            "metrics": base_metrics
        },
        "dilated": {
            "parameters": dil_params,
            "metrics": dil_metrics
        },
        "delta": {
            "miou_gain": round((dil_metrics["miou"] - base_metrics["miou"]) * 100, 2),
            "pedestrian_iou_gain": round((dil_metrics["per_class_iou"]["class_1"] - base_metrics["per_class_iou"]["class_1"]) * 100, 2)
        }
    }

    json_path = os.path.join(results_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results

if __name__ == "__main__":
    run_evaluation()
