"""
Results Tracker for Experiment Management

Manages centralized experiment registry and results storage.
"""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ResultsTracker:
    """Tracks experiment results with centralized registry."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize results tracker.

        Args:
            results_dir: Base directory for results storage
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.results_dir / "experiments.csv"
        self._initialize_registry()

    def _initialize_registry(self):
        """Create experiments registry CSV if it doesn't exist."""
        if not self.registry_path.exists():
            with open(self.registry_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "exp_id",
                    "name",
                    "start_time",
                    "end_time",
                    "status",
                    "steps",
                    "duration_hours",
                    "final_score",
                    "best_score",
                    "worst_score",
                    "config_path",
                ])

    def register_experiment(
        self,
        exp_id: str,
        name: str,
        config_path: str,
        start_time: Optional[datetime] = None,
    ):
        """
        Register a new experiment in the registry.

        Args:
            exp_id: Unique experiment identifier
            name: Experiment name
            config_path: Path to experiment config file
            start_time: Experiment start time (defaults to now)
        """
        if start_time is None:
            start_time = datetime.now()

        # Create experiment results directory
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Copy config to results directory
        if Path(config_path).exists():
            shutil.copy(config_path, exp_dir / "config.yaml")

        # Add to registry
        with open(self.registry_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                exp_id,
                name,
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "",  # end_time - to be filled on completion
                "running",
                0,  # steps - to be updated
                0.0,  # duration_hours
                "",  # final_score
                "",  # best_score
                "",  # worst_score
                config_path,
            ])

        print(f"✓ Registered experiment: {exp_id}")

    def finalize_experiment(
        self,
        exp_id: str,
        steps: int,
        duration_hours: float,
        final_score: float,
        best_score: float,
        worst_score: float,
    ):
        """
        Update registry with final experiment results.

        Args:
            exp_id: Experiment identifier
            steps: Total training steps completed
            duration_hours: Training duration in hours
            final_score: Final mean score
            best_score: Best score achieved
            worst_score: Worst score in final population
        """
        # Read registry
        rows = []
        with open(self.registry_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Update experiment row
        for row in rows:
            if row["exp_id"] == exp_id:
                row["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row["status"] = "completed"
                row["steps"] = str(steps)
                row["duration_hours"] = f"{duration_hours:.2f}"
                row["final_score"] = f"{final_score:.2f}"
                row["best_score"] = f"{best_score:.2f}"
                row["worst_score"] = f"{worst_score:.2f}"
                break

        # Write back
        with open(self.registry_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Finalized experiment: {exp_id}")

    def save_metrics_json(
        self,
        exp_id: str,
        metrics: Dict[str, Any],
    ):
        """
        Save detailed metrics as JSON.

        Args:
            exp_id: Experiment identifier
            metrics: Dictionary of metrics to save
        """
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Saved metrics: {metrics_path}")

    def save_training_artifacts(
        self,
        exp_id: str,
        model_path: str,
        scores_plot_path: Optional[str] = None,
        scores_data_path: Optional[str] = None,
    ):
        """
        Copy training artifacts to results directory.

        Args:
            exp_id: Experiment identifier
            model_path: Path to trained model
            scores_plot_path: Path to training scores plot
            scores_data_path: Path to scores numpy array
        """
        exp_dir = self.results_dir / exp_id

        # Copy model
        if Path(model_path).exists():
            shutil.copy(model_path, exp_dir / Path(model_path).name)

        # Copy plot
        if scores_plot_path and Path(scores_plot_path).exists():
            shutil.copy(scores_plot_path, exp_dir / Path(scores_plot_path).name)

        # Copy data
        if scores_data_path and Path(scores_data_path).exists():
            shutil.copy(scores_data_path, exp_dir / Path(scores_data_path).name)

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data from registry.

        Args:
            exp_id: Experiment identifier

        Returns:
            Dictionary of experiment data, or None if not found
        """
        with open(self.registry_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["exp_id"] == exp_id:
                    return dict(row)
        return None

    def list_experiments(self, status: Optional[str] = None) -> list:
        """
        List all experiments, optionally filtered by status.

        Args:
            status: Filter by status ('running', 'completed', 'failed'), or None for all

        Returns:
            List of experiment dictionaries
        """
        with open(self.registry_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if status:
            rows = [r for r in rows if r["status"] == status]

        return rows
