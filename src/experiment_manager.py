"""
Experiment Manager for YAML-based Configuration

Loads and validates experiment configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter


class ExperimentConfig:
    """Container for experiment configuration."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from config dictionary."""
        self.raw = config_dict
        self.name = config_dict.get("name", "unnamed")
        self.description = config_dict.get("description", "")
        self.seed = config_dict.get("seed", 42)
        self.training = config_dict.get("training", {})
        self.hyperparameters = config_dict.get("hyperparameters", {})
        self.network = config_dict.get("network", {})
        self.hpo_config = config_dict.get("hpo_config", {})
        self.mutation = config_dict.get("mutation", {})


class ExperimentManager:
    """Manages experiment configuration loading and validation."""

    def __init__(self, config_path: str):
        """
        Initialize experiment manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: ExperimentConfig = None

    def load_config(self) -> ExperimentConfig:
        """
        Load and validate YAML configuration.

        Returns:
            ExperimentConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config is invalid YAML
            ValueError: If required fields are missing
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.config = ExperimentConfig(config_dict)
        self._validate_config()
        return self.config

    def _validate_config(self):
        """Validate that required configuration fields are present."""
        required_fields = {
            "training": ["max_steps", "num_envs", "evo_steps"],
            "hyperparameters": ["population_size", "algo"],  # Algorithm-agnostic
            "network": ["latent_dim"],
        }

        for section, fields in required_fields.items():
            section_config = getattr(self.config, section, {})
            for field in fields:
                if field not in section_config:
                    raise ValueError(f"Missing required field '{field}' in section '{section}'")

    def get_init_hp(self) -> Dict[str, Any]:
        """
        Build INIT_HP dictionary from config (algorithm-agnostic).

        Returns:
            Dictionary suitable for AgileRL's create_population
        """
        hp = self.config.hyperparameters

        # Convert all hyperparameters to uppercase keys for AgileRL
        init_hp = {}
        for key, value in hp.items():
            # Convert key to uppercase
            init_hp[key.upper()] = value

        return init_hp

    def get_net_config(self) -> Dict[str, Any]:
        """
        Build NET_CONFIG dictionary from config.

        Returns:
            Dictionary suitable for AgileRL's create_population
        """
        net = self.config.network

        # Get hidden sizes and latent dim
        encoder_hidden = net.get("encoder_hidden_size", [64])
        head_hidden = net.get("head_hidden_size", [64])
        latent_dim = net["latent_dim"]

        # Set max_hidden_layers to accommodate the config
        # Allow up to 10 layers by default, or the config size + buffer
        encoder_max = max(10, len(encoder_hidden) + 5)
        head_max = max(10, len(head_hidden) + 5)

        # Set max_latent_dim to accommodate mutations
        # Allow up to 512 or current latent_dim * 2
        max_latent = max(512, latent_dim * 2)

        return {
            "latent_dim": latent_dim,
            "max_latent_dim": max_latent,
            "encoder_config": {
                "hidden_size": encoder_hidden,
                "max_hidden_layers": encoder_max,
            },
            "head_config": {
                "hidden_size": head_hidden,
                "max_hidden_layers": head_max,
            },
        }

    def get_hpo_config(self) -> HyperparameterConfig:
        """
        Build HyperparameterConfig from config.

        Returns:
            HyperparameterConfig for evolutionary HPO
        """
        hpo = self.config.hpo_config

        params = {}

        # Learning rates
        if "lr_actor" in hpo:
            params["lr_actor"] = RLParameter(**hpo["lr_actor"])
        if "lr_critic" in hpo:
            params["lr_critic"] = RLParameter(**hpo["lr_critic"])

        # Batch size
        if "batch_size" in hpo:
            bs_config = hpo["batch_size"].copy()
            if "dtype" in bs_config and bs_config["dtype"] == "int":
                bs_config["dtype"] = int
            params["batch_size"] = RLParameter(**bs_config)

        # Learn step
        if "learn_step" in hpo:
            ls_config = hpo["learn_step"].copy()
            if "dtype" in ls_config and ls_config["dtype"] == "int":
                ls_config["dtype"] = int
            params["learn_step"] = RLParameter(**ls_config)

        return HyperparameterConfig(**params)

    def get_mutation_config(self) -> Dict[str, Any]:
        """
        Get mutation configuration.

        Returns:
            Dictionary with mutation probabilities and parameters
        """
        mut = self.config.mutation

        return {
            "no_mutation": mut.get("no_mutation", 0.2),
            "architecture": mut.get("architecture", 0.2),
            "new_layer_prob": mut.get("new_layer", 0.2),
            "parameters": mut.get("parameter", 0.2),
            "activation": mut.get("activation", 0.0),
            "rl_hp": mut.get("rl_hp", 0.2),
            "mutation_sd": mut.get("mutation_sd", 0.1),
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.

        Returns:
            Dictionary with training parameters
        """
        return {
            "max_steps": self.config.training["max_steps"],
            "num_envs": self.config.training["num_envs"],
            "evo_steps": self.config.training["evo_steps"],
            "checkpoint_interval": self.config.training.get("checkpoint_interval", 100000),
            "learning_delay": self.config.training.get("learning_delay", 0),
            "eval_steps": self.config.training.get("eval_steps", None),
            "eval_loop": self.config.training.get("eval_loop", 1),
        }
