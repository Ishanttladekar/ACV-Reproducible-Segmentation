import sys
import time
from experiments.train import train_experiment
from experiments.evaluate import run_evaluation
from experiments.benchmark import run_benchmark
from experiments.visualize import plot_training_curves, generate_qualitative_comparisons, generate_failure_cases

def main():
    total_start = time.time()
    print("=" * 60, flush=True)
    print("Running Reproducible Segmentation Experiments", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Training Baseline Model ---", flush=True)
    train_experiment(model_type="baseline", epochs=5, batch_size=8, seed=42)

    print("\n--- Training Dilated Model ---", flush=True)
    train_experiment(model_type="dilated", epochs=5, batch_size=8, seed=42)

    print("\n--- Evaluating Test Set ---", flush=True)
    run_evaluation()

    print("\n--- Benchmarking Latency ---", flush=True)
    run_benchmark()

    print("\n--- Generating Visualizations ---", flush=True)
    plot_training_curves()
    generate_qualitative_comparisons()
    generate_failure_cases()

    print(f"\nExecution finished in {time.time() - total_start:.2f}s", flush=True)

if __name__ == "__main__":
    main()
