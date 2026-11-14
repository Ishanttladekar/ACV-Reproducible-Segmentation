# Controlled Empirical Evaluation of Dilated Convolutions in Modest Encoder-Decoder Architectures for Semantic Segmentation

## Overview
This repository contains the complete, reproducible implementation for Assignment B (CS107203CS Unit 3: Advanced Computer Vision). The experiment establishes a controlled empirical comparison between a modest 4-stage encoder-decoder baseline and a dilated (atrous) bottleneck variant on the Penn-Fudan Pedestrian Database.

## Directory Structure
```
ACV/
├── dataset/
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── baseline_unet.py
│   ├── dilated_unet.py
│   └── metrics.py
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── visualize.py
├── results/
│   ├── baseline/
│   ├── dilated/
│   ├── comparative_metrics.csv
│   ├── latency_benchmark.csv
│   ├── test_metrics.json
│   ├── latency_benchmark.json
│   ├── training_curves.png
│   ├── qualitative_comparisons.png
│   └── failure_cases_analysis.png
├── report/
│   └── <ROLL_NO>_Report.md
├── requirements.txt
├── run_all_experiments.py
└── README.md
```

## Environment Specifications
- Python: 3.11.9
- PyTorch: 2.15.0+cpu
- Torchvision: 0.29.2+cpu
- Platform: Windows 10 (10.0.26200), GenuineIntel x86_64, 6 Threads

## Installation
```bash
pip install -r requirements.txt
```

## Running the Complete Pipeline
To execute training of both models, test set evaluation, latency benchmarking, and visualization generation:
```bash
python run_all_experiments.py
```

Or run individual stages:
```bash
# 1. Train models
python experiments/train.py --model baseline --epochs 5
python experiments/train.py --model dilated --epochs 5

# 2. Evaluate held-out test split
python experiments/evaluate.py

# 3. Benchmark latency and throughput
python experiments/benchmark.py

# 4. Generate visual figures
python experiments/visualize.py
```

## Empirical Summary
| Model | Params | mIoU (%) | Pedestrian IoU (%) | Mean Dice (%) | Pixel Acc (%) | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| Baseline U-Net (d=1) | 1,797,794 | 73.85 | 57.03 | 83.87 | 91.71 | 83.54 +/- 12.07 | 11.97 |
| Dilated U-Net (d=2, 4) | 1,797,794 | 75.50 | 59.43 | 85.08 | 92.51 | 85.99 +/- 9.91 | 11.63 |
| **Delta** | **0.00%** | **+1.65%** | **+2.40%** | **+1.21%** | **+0.80%** | **+2.45 ms** | **-0.34 FPS** |
