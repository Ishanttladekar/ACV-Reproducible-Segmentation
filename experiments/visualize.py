import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from dataset.dataset import PennFudanPedDataset
from models.baseline_unet import BaselineUNet
from models.dilated_unet import DilatedUNet

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = tensor.cpu().numpy() * std + mean
    img_np = np.clip(img_np, 0.0, 1.0)
    return np.transpose(img_np, (1, 2, 0))

def plot_training_curves(results_dir="d:/ACV/results"):
    base_log_path = os.path.join(results_dir, "baseline", "training_log.csv")
    dil_log_path = os.path.join(results_dir, "dilated", "training_log.csv")

    if not os.path.exists(base_log_path) or not os.path.exists(dil_log_path):
        return

    df_base = pd.read_csv(base_log_path)
    df_dil = pd.read_csv(dil_log_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(df_base["epoch"], df_base["train_loss"], "b--", label="Baseline Train", alpha=0.8)
    axes[0].plot(df_base["epoch"], df_base["val_loss"], "b-", label="Baseline Val", linewidth=2)
    axes[0].plot(df_dil["epoch"], df_dil["train_loss"], "r--", label="Dilated Train", alpha=0.8)
    axes[0].plot(df_dil["epoch"], df_dil["val_loss"], "r-", label="Dilated Val", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training & Validation Loss", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(df_base["epoch"], df_base["val_miou"] * 100, "b-o", label="Baseline Val mIoU", linewidth=2)
    axes[1].plot(df_dil["epoch"], df_dil["val_miou"] * 100, "r-s", label="Dilated Val mIoU", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation mIoU (%)", fontsize=11)
    axes[1].set_title("Validation mIoU Comparison", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def generate_qualitative_comparisons(data_dir="d:/ACV/data/PennFudanPed",
                                     results_dir="d:/ACV/results",
                                     num_samples=5,
                                     seed=42,
                                     img_size=(128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PennFudanPedDataset(data_dir, split="test", seed=seed, img_size=img_size)

    base_model = BaselineUNet()
    base_model.load_state_dict(torch.load(os.path.join(results_dir, "baseline", "best_model.pth"), map_location=device))
    base_model.to(device).eval()

    dil_model = DilatedUNet()
    dil_model.load_state_dict(torch.load(os.path.join(results_dir, "dilated", "best_model.pth"), map_location=device))
    dil_model.to(device).eval()

    fig, axes = plt.subplots(num_samples, 5, figsize=(16, 3.2 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)

    col_titles = ["Input Image", "Ground Truth", "Baseline Prediction", "Dilated Prediction", "Error Map"]

    for i in range(num_samples):
        sample = dataset[i]
        img_t = sample["image"].unsqueeze(0).to(device)
        mask_t = sample["mask"].cpu().numpy()

        with torch.no_grad():
            out_base = torch.argmax(base_model(img_t), dim=1).squeeze(0).cpu().numpy()
            out_dil = torch.argmax(dil_model(img_t), dim=1).squeeze(0).cpu().numpy()

        rgb_disp = denormalize_image(sample["image"])

        error_map = np.zeros((*mask_t.shape, 3), dtype=np.float32)
        error_map[mask_t == out_dil] = 0.85
        error_map[(mask_t == 0) & (out_dil == 1)] = [1.0, 0.15, 0.15]
        error_map[(mask_t == 1) & (out_dil == 0)] = [0.15, 0.45, 1.0]

        axes[i, 0].imshow(rgb_disp)
        axes[i, 1].imshow(mask_t, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(out_base, cmap="viridis", vmin=0, vmax=1)
        axes[i, 3].imshow(out_dil, cmap="viridis", vmin=0, vmax=1)
        axes[i, 4].imshow(error_map)

        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11, fontweight="bold", pad=8)

        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=10, fontweight="bold")

    comp_path = os.path.join(results_dir, "qualitative_comparisons.png")
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close()

def generate_failure_cases(data_dir="d:/ACV/data/PennFudanPed",
                           results_dir="d:/ACV/results",
                           seed=42,
                           img_size=(128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PennFudanPedDataset(data_dir, split="test", seed=seed, img_size=img_size)

    dil_model = DilatedUNet()
    dil_model.load_state_dict(torch.load(os.path.join(results_dir, "dilated", "best_model.pth"), map_location=device))
    dil_model.to(device).eval()

    sample_scores = []
    for i in range(len(dataset)):
        sample = dataset[i]
        img_t = sample["image"].unsqueeze(0).to(device)
        mask_t = sample["mask"].cpu().numpy()

        with torch.no_grad():
            out_dil = torch.argmax(dil_model(img_t), dim=1).squeeze(0).cpu().numpy()

        intersection = np.logical_and(mask_t == 1, out_dil == 1).sum()
        union = np.logical_or(mask_t == 1, out_dil == 1).sum()
        ped_iou = (intersection / union) if union > 0 else 1.0
        sample_scores.append((i, ped_iou, sample, out_dil))

    sample_scores.sort(key=lambda x: x[1])
    worst_samples = sample_scores[:3]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)

    titles = ["Input Image", "Ground Truth", "Prediction", "Error Map"]

    for row, (idx, iou, sample, pred) in enumerate(worst_samples):
        rgb_disp = denormalize_image(sample["image"])
        mask_t = sample["mask"].cpu().numpy()

        error_map = np.zeros((*mask_t.shape, 3), dtype=np.float32)
        error_map[mask_t == pred] = 0.85
        error_map[(mask_t == 0) & (pred == 1)] = [1.0, 0.2, 0.2]
        error_map[(mask_t == 1) & (pred == 0)] = [0.2, 0.4, 1.0]

        axes[row, 0].imshow(rgb_disp)
        axes[row, 1].imshow(mask_t, cmap="gray")
        axes[row, 2].imshow(pred, cmap="viridis")
        axes[row, 3].imshow(error_map)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=11, fontweight="bold", pad=8)

        axes[row, 0].set_ylabel(f"Failure #{row+1}\nIoU: {iou*100:.1f}%", fontsize=10, fontweight="bold")

    failure_path = os.path.join(results_dir, "failure_cases_analysis.png")
    plt.savefig(failure_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_training_curves()
    generate_qualitative_comparisons()
    generate_failure_cases()
