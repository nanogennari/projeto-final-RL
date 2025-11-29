#!/usr/bin/env python3
"""
Script to visualize training progress for completed experiments.
Generates a plot showing 10-iteration rolling averages for all completed experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_experiment_data(results_dir: Path, exp_id: str) -> np.ndarray:
    """Load training data for a specific experiment."""
    data_file = results_dir / exp_id / f"{exp_id}_data.npy"
    if data_file.exists():
        return np.load(data_file)
    return None


def rolling_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Calculate rolling average with specified window size."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def plot_experiments(results_dir: Path = Path("results"),
                     output_file: str = "training_progress.png",
                     window: int = 10,
                     target_score: float = -60.0):
    """
    Plot training progress for all completed experiments.

    Args:
        results_dir: Directory containing experiment results
        output_file: Output filename for the plot
        window: Rolling average window size
        target_score: Target score to show as horizontal line
    """
    # Read experiments metadata
    experiments_csv = results_dir / "experiments.csv"
    df = pd.read_csv(experiments_csv)

    # Filter completed experiments only
    completed = df[df['status'] == 'completed'].copy()

    if len(completed) == 0:
        print("No completed experiments found.")
        return

    print(f"Found {len(completed)} completed experiments")

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot each experiment
    for _, exp in completed.iterrows():
        exp_id = exp['exp_id']
        name = exp['name']
        best_score = exp['best_score']

        # Load training data
        data = load_experiment_data(results_dir, exp_id)
        if data is None:
            print(f"Warning: No data found for {exp_id}")
            continue

        # Calculate rolling average
        smoothed = rolling_average(data, window=window)

        # Plot with best score in legend
        iterations = np.arange(len(smoothed))
        label = f"{name}"
        plt.plot(iterations, smoothed, label=label, linewidth=2, alpha=0.8)

    # Add target line
    plt.axhline(y=target_score, color='green', linestyle='--',
                linewidth=2, label=f'Target ({target_score})', alpha=0.7)

    # Formatting
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'Training Progress - {window}-Iteration Rolling Average', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot experiment training progress')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing experiment results (default: results)')
    parser.add_argument('--output', type=str, default='training_progress.png',
                       help='Output filename (default: training_progress.png)')
    parser.add_argument('--window', type=int, default=10,
                       help='Rolling average window size (default: 10)')
    parser.add_argument('--target', type=float, default=-60.0,
                       help='Target score line (default: -60.0)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return

    plot_experiments(
        results_dir=results_dir,
        output_file=args.output,
        window=args.window,
        target_score=args.target
    )


if __name__ == '__main__':
    main()
