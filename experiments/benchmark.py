import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import platform
import numpy as np
import pandas as pd
import torch

from models.baseline_unet import BaselineUNet
from models.dilated_unet import DilatedUNet
from models.metrics import count_parameters

def benchmark_model(model: torch.nn.Module, device: torch.device,
                    input_shape=(1, 3, 128, 128), warmup=20, runs=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat)

    param_dict = count_parameters(model)

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 2),
        "total_parameters": param_dict["total_parameters"],
        "warmup_runs": warmup,
        "benchmark_runs": runs,
        "batch_size": input_shape[0],
        "input_resolution": f"{input_shape[2]}x{input_shape[3]}"
    }

def run_benchmark(output_dir="d:/ACV/results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print(f"Latency & Resource Benchmark on {device}", flush=True)
    print(f"Platform: {platform.platform()} | Processor: {platform.processor()}", flush=True)
    print("=" * 60, flush=True)

    baseline = BaselineUNet()
    dilated = DilatedUNet()

    base_results = benchmark_model(baseline, device)
    dil_results = benchmark_model(dilated, device)

    rows = [
        {
            "Architecture": "Baseline U-Net (d=1)",
            "Parameters": base_results["total_parameters"],
            "Mean Latency (ms)": f"{base_results['mean_latency_ms']} +/- {base_results['std_latency_ms']}",
            "Median Latency (ms)": base_results["median_latency_ms"],
            "P95 Latency (ms)": base_results["p95_latency_ms"],
            "Throughput (FPS)": base_results["throughput_fps"]
        },
        {
            "Architecture": "Dilated U-Net (d=2, 4)",
            "Parameters": dil_results["total_parameters"],
            "Mean Latency (ms)": f"{dil_results['mean_latency_ms']} +/- {dil_results['std_latency_ms']}",
            "Median Latency (ms)": dil_results["median_latency_ms"],
            "P95 Latency (ms)": dil_results["p95_latency_ms"],
            "Throughput (FPS)": dil_results["throughput_fps"]
        }
    ]

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "latency_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print("\nLatency Benchmark:", flush=True)
    print(df.to_string(index=False), flush=True)

    full_report = {
        "conditions": {
            "device": str(device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_threads": torch.get_num_threads(),
            "input_resolution": "128x128",
            "batch_size": 1,
            "warmup_runs": 20,
            "benchmark_runs": 100
        },
        "baseline": base_results,
        "dilated": dil_results
    }

    json_path = os.path.join(output_dir, "latency_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    return full_report

if __name__ == "__main__":
    run_benchmark()
