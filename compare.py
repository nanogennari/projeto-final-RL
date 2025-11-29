#!/usr/bin/env python3
"""
Compare results from multiple experiments.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def load_experiment_data(results_dir: Path, exp_id: str) -> np.ndarray:
    """Load training data for a specific experiment."""
    data_file = results_dir / exp_id / f"{exp_id}_data.npy"
    if data_file.exists():
        return np.load(data_file)
    return None


def list_experiments(results_dir: Path = Path("results")):
    """List all completed experiments."""
    experiments_csv = results_dir / "experiments.csv"
    if not experiments_csv.exists():
        print("No experiments found.")
        return

    df = pd.read_csv(experiments_csv)
    completed = df[df['status'] == 'completed'].copy()

    if len(completed) == 0:
        print("No completed experiments found.")
        return

    print("\nCompleted Experiments:")
    print("=" * 100)
    print(f"{'ID':<25} {'Name':<30} {'Best Score':<12} {'Steps':<12} {'Duration':<10}")
    print("-" * 100)

    for _, exp in completed.iterrows():
        exp_id = exp['exp_id']
        name = exp['name']
        best = exp['best_score']
        steps = exp['steps']
        duration = exp['duration_hours']
        print(f"{exp_id:<25} {name:<30} {best:<12.2f} {steps:<12} {duration:<10.2f}h")

    print("=" * 100)
    print(f"\nTotal: {len(completed)} experiments")


def compare_experiments(
    exp_ids: List[str],
    results_dir: Path = Path("results"),
    output_file: str = None
):
    """Compare multiple experiments."""
    experiments_csv = results_dir / "experiments.csv"
    if not experiments_csv.exists():
        print("No experiments registry found.")
        return

    df = pd.read_csv(experiments_csv)

    # Filter to requested experiments
    experiments = df[df['exp_id'].isin(exp_ids)]

    if len(experiments) == 0:
        print("No matching experiments found.")
        return

    # Print comparison table
    print("\nExperiment Comparison:")
    print("=" * 120)
    print(f"{'Experiment':<25} {'Name':<25} {'Final':<10} {'Best':<10} {'Worst':<10} {'Steps':<12} {'Duration':<10}")
    print("-" * 120)

    for _, exp in experiments.iterrows():
        exp_id = exp['exp_id']
        name = exp['name']
        final = exp['final_score'] if pd.notna(exp['final_score']) else 'N/A'
        best = exp['best_score'] if pd.notna(exp['best_score']) else 'N/A'
        worst = exp['worst_score'] if pd.notna(exp['worst_score']) else 'N/A'
        steps = exp['steps']
        duration = exp['duration_hours']

        final_str = f"{final:.2f}" if final != 'N/A' else final
        best_str = f"{best:.2f}" if best != 'N/A' else best
        worst_str = f"{worst:.2f}" if worst != 'N/A' else worst

        print(f"{exp_id:<25} {name:<25} {final_str:<10} {best_str:<10} {worst_str:<10} {steps:<12} {duration:<10.2f}h")

    print("=" * 120)

    # Plot training curves
    plt.figure(figsize=(14, 7))

    for _, exp in experiments.iterrows():
        exp_id = exp['exp_id']
        name = exp['name']

        data = load_experiment_data(results_dir, exp_id)
        if data is None:
            print(f"Warning: No training data found for {exp_id}")
            continue

        # Plot with 10-iteration rolling average
        smoothed = pd.Series(data).rolling(window=10, min_periods=1).mean().values
        iterations = np.arange(len(smoothed))
        plt.plot(iterations, smoothed, label=name, linewidth=2, alpha=0.8)

    plt.axhline(y=-60, color='green', linestyle='--', linewidth=2, label='Target (-60)', alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Training Progress Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_file}")
    else:
        plt.savefig("comparison.png", dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: comparison.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python compare.py --list

  # Compare two experiments
  python compare.py exp_20251128_123149 exp_20251128_134031

  # Compare multiple experiments with custom output
  python compare.py exp_001 exp_002 exp_003 --output my_comparison.png
        """
    )

    parser.add_argument('exp_ids', nargs='*', help='Experiment IDs to compare')
    parser.add_argument('--list', action='store_true', help='List all completed experiments')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing experiment results (default: results)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for comparison plot (default: comparison.png)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return

    if args.list:
        list_experiments(results_dir)
    elif len(args.exp_ids) > 0:
        compare_experiments(args.exp_ids, results_dir, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
