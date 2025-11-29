"""
Progress Tracker for Live Training Monitoring

Provides real-time CSV updates during training for easy monitoring.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class ProgressTracker:
    """Tracks training progress with live CSV updates."""

    def __init__(self, experiment_id: str, progress_dir: str = "progress", max_steps: int = 2_000_000):
        """
        Initialize progress tracker.

        Args:
            experiment_id: Unique identifier for this experiment
            progress_dir: Directory to store progress CSV files
            max_steps: Maximum training steps for ETA calculation
        """
        self.experiment_id = experiment_id
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.progress_dir / "current_run.csv"
        self.max_steps = max_steps
        self.start_time = None

        # Initialize CSV file
        self._initialize_csv()

    def _initialize_csv(self):
        """Create CSV file with headers."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "experiment_id",
                "step",
                "elapsed_sec",
                "mean_score",
                "best_score",
                "worst_score",
                "fitness_avg",
                "elite_mutations",
                "pct_complete",
                "eta_hours",
            ])

    def update(
        self,
        step: int,
        elapsed_time: float,
        mean_score: float,
        best_score: float,
        worst_score: float,
        fitness_avg: float,
        elite_mutations: int,
    ):
        """
        Append progress update to CSV.

        Args:
            step: Current training step
            elapsed_time: Elapsed time in seconds
            mean_score: Mean population score
            best_score: Best score in population
            worst_score: Worst score in population
            fitness_avg: Average fitness (5-iteration rolling)
            elite_mutations: Number of mutations on elite agent
        """
        # Calculate progress metrics
        pct_complete = (step / self.max_steps) * 100

        # Estimate time remaining
        if step > 0 and elapsed_time > 0:
            steps_per_sec = step / elapsed_time
            remaining_steps = self.max_steps - step
            eta_hours = (remaining_steps / steps_per_sec) / 3600 if steps_per_sec > 0 else 0
        else:
            eta_hours = 0

        # Write to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.experiment_id,
                step,
                f"{elapsed_time:.1f}",
                f"{mean_score:.2f}",
                f"{best_score:.2f}",
                f"{worst_score:.2f}",
                f"{fitness_avg:.2f}",
                elite_mutations,
                f"{pct_complete:.2f}",
                f"{eta_hours:.2f}",
            ])

    def get_latest_metrics(self, n_rows: int = 10) -> list:
        """
        Read last N rows from progress CSV.

        Args:
            n_rows: Number of recent rows to return

        Returns:
            List of row dictionaries
        """
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[-n_rows:] if len(rows) > n_rows else rows

    def get_current_status(self) -> Optional[dict]:
        """
        Get most recent progress update.

        Returns:
            Dictionary with latest metrics, or None if no data
        """
        latest = self.get_latest_metrics(n_rows=1)
        return latest[0] if latest else None
