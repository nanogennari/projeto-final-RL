#!/usr/bin/env python3
"""
Training Progress Summary

Displays formatted summary of current training run.
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    return str(timedelta(seconds=int(seconds)))


def get_trend(values: list, threshold: float = 0.5) -> str:
    """Determine trend from list of values."""
    if len(values) < 2:
        return "→"

    recent = values[-5:] if len(values) >= 5 else values
    if len(recent) < 2:
        return "→"

    change = recent[-1] - recent[0]
    if abs(change) < threshold:
        return "→"  # Stable
    elif change > 0:
        return "▲"  # Improving (scores are negative, so increasing means getting better)
    else:
        return "▼"  # Declining


def display_summary():
    """Display formatted training progress summary."""
    progress_file = Path("progress/current_run.csv")

    if not progress_file.exists():
        print("No training in progress. Start training first.")
        return

    # Read progress data
    with open(progress_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No progress data available yet.")
        return

    # Get latest and recent rows
    latest = rows[-1]
    recent = rows[-10:] if len(rows) >= 10 else rows

    # Extract metrics
    experiment_id = latest["experiment_id"]
    step = int(latest["step"])
    max_steps = 2_000_000  # Default, could be read from config
    elapsed_sec = float(latest["elapsed_sec"])
    pct_complete = float(latest["pct_complete"])
    eta_hours = float(latest["eta_hours"])

    # Recent performance
    mean_scores = [float(r["mean_score"]) for r in recent]
    best_scores = [float(r["best_score"]) for r in recent]
    fitness_avgs = [float(r["fitness_avg"]) for r in recent]

    # Calculate trends
    mean_trend = get_trend(mean_scores)
    best_trend = get_trend(best_scores)

    # Check for checkpoint
    checkpoint_dir = Path(f"checkpoints/{experiment_id}")
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            last_checkpoint_step = int(checkpoints[-1].stem.split("_")[1])
            checkpoint_info = f"Step {last_checkpoint_step:,}"
        else:
            checkpoint_info = "None"
    else:
        checkpoint_info = "None"

    # Display summary
    print("=" * 60)
    print("           EXPERIMENT PROGRESS SUMMARY")
    print("=" * 60)
    print()
    print(f"Experiment:    {experiment_id}")
    print(f"Status:        Training")
    print()
    print(f"Steps:         {step:,} / {max_steps:,}  ({pct_complete:.1f}%)")
    print(f"Elapsed:       {format_duration(elapsed_sec)}")
    print(f"ETA:           {format_duration(eta_hours * 3600)}")
    print()
    print("-" * 60)
    print("RECENT PERFORMANCE (last 10 evolution cycles)")
    print("-" * 60)
    print()
    print(f"Mean Score:    {mean_scores[-1]:>7.2f}  {mean_trend}")
    print(f"Best Score:    {best_scores[-1]:>7.2f}  {best_trend}")
    print(f"Fitness Avg:   {fitness_avgs[-1]:>7.2f}")
    print()
    if len(mean_scores) >= 5:
        improvement = mean_scores[-1] - mean_scores[-5]
        print(f"Improvement (last 5 cycles): {improvement:+.2f}")
    print()
    print("-" * 60)
    print("CHECKPOINTS")
    print("-" * 60)
    print()
    print(f"Last Checkpoint:  {checkpoint_info}")
    print()
    print("=" * 60)

    # Show recent history table
    print()
    print("RECENT HISTORY:")
    print()
    print(f"{'Step':<10} {'Elapsed':<10} {'Mean':<10} {'Best':<10} {'Fitness':<10}")
    print("-" * 60)
    for row in recent[-5:]:
        step_val = int(row["step"])
        elapsed_val = format_duration(float(row["elapsed_sec"]))
        mean_val = float(row["mean_score"])
        best_val = float(row["best_score"])
        fitness_val = float(row["fitness_avg"])

        print(f"{step_val:<10,} {elapsed_val:<10} {mean_val:<10.2f} {best_val:<10.2f} {fitness_val:<10.2f}")

    print()
    print("=" * 60)
    print()
    print("TIP: Monitor live with: tail -f progress/current_run.csv")
    print()


if __name__ == "__main__":
    try:
        display_summary()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
