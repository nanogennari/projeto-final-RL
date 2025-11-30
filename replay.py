import os
import argparse
import pandas as pd
import imageio
import numpy as np
import torch
import json
from mpe2 import simple_speaker_listener_v4
from PIL import Image, ImageDraw

from agilerl.algorithms import MATD3
from smpe import SMPE


# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


def get_latest_experiment():
    """Get the latest experiment ID from experiments.csv"""
    try:
        df = pd.read_csv('results/experiments.csv')
        if not df.empty:
            # Get the last exp_id from the CSV
            last_exp_id = df.iloc[-1]['exp_id']
            return last_exp_id
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        pass
    return None


def detect_algorithm(exp_id: str) -> str:
    """
    Detect algorithm type from experiment metadata.

    Detection priority:
    1. Check metrics.json for ALGO field
    2. Check experiments.csv for algorithm pattern in name
    3. Introspect checkpoint for SMPE-specific keys

    Returns:
        Algorithm name ("MATD3" or "SMPE")

    Raises:
        ValueError: If algorithm cannot be determined
    """
    # Priority 1: Check metrics.json
    metrics_path = f"results/{exp_id}/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                algo = metrics.get('hyperparameters', {}).get('ALGO')
                if algo:
                    return algo.upper()
        except (json.JSONDecodeError, KeyError):
            pass

    # Priority 2: Check experiments.csv
    csv_path = "results/experiments.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            exp_row = df[df['exp_id'] == exp_id]
            if not exp_row.empty:
                name = str(exp_row.iloc[0]['name']).lower()
                if 'smpe' in name:
                    return 'SMPE'
                elif 'matd3' in name or 'maddpg' in name:
                    return 'MATD3'
        except (pd.errors.EmptyDataError, KeyError):
            pass

    # Priority 3: Checkpoint introspection
    model_path = f"results/{exp_id}/{exp_id}_model.pt"
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'hyperparameters' in ckpt:
                if 'BELIEF_LATENT_DIM' in ckpt['hyperparameters']:
                    return 'SMPE'
                return 'MATD3'
        except Exception:
            pass

    raise ValueError(f"Cannot detect algorithm for experiment {exp_id}")


def create_environment(algorithm: str, render_mode: str = "rgb_array"):
    """
    Create environment configured for specific algorithm.

    Args:
        algorithm: Algorithm name ("MATD3" or "SMPE")
        render_mode: Rendering mode for environment

    Returns:
        Configured PettingZoo parallel environment
    """
    continuous = (algorithm == "MATD3")
    env = simple_speaker_listener_v4.parallel_env(
        continuous_actions=continuous,
        render_mode=render_mode
    )
    return env


# ============ Model Loading Factory Pattern ============


class ModelLoader:
    """Base class for algorithm-specific model loading."""

    @staticmethod
    def load(model_path: str, env, agent_ids: list, device: torch.device):
        """Load model from checkpoint."""
        raise NotImplementedError

    @staticmethod
    def prepare_for_inference(model):
        """Prepare model for inference (disable training mode, etc.)."""
        raise NotImplementedError


class MATD3Loader(ModelLoader):
    """Loader for MATD3 models."""

    @staticmethod
    def load(model_path: str, env, agent_ids: list, device: torch.device):
        """Load MATD3 model from checkpoint."""
        try:
            # Try AgileRL's built-in load method
            model = MATD3.load(model_path, device)
            print(f"Loaded MATD3 via MATD3.load() from: {model_path}")
            return model
        except Exception as e:
            print(f"MATD3.load() failed: {e}. Trying manual load...")

            # Fallback: manual reconstruction
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            if not isinstance(ckpt, dict) or "network_info" not in ckpt:
                raise ValueError(f"Invalid MATD3 checkpoint format: {model_path}")

            observation_spaces = [env.observation_space(a) for a in agent_ids]
            action_spaces = [env.action_space(a) for a in agent_ids]

            model = MATD3(
                observation_spaces=observation_spaces,
                action_spaces=action_spaces,
                agent_ids=agent_ids,
                device=device
            )
            model.load_state_dict(ckpt)
            print(f"Loaded MATD3 state dict into new instance from: {model_path}")
            return model

    @staticmethod
    def prepare_for_inference(model):
        """MATD3 needs no special preparation for inference."""
        return model


class SMPELoader(ModelLoader):
    """Loader for SMPE models."""

    @staticmethod
    def load(model_path: str, env, agent_ids: list, device: torch.device):
        """Load SMPE model with all required initialization."""
        # Load hyperparameters from metrics.json
        exp_id = os.path.basename(os.path.dirname(model_path))
        metrics_path = f"results/{exp_id}/metrics.json"

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"SMPE requires metrics.json to load hyperparameters: {metrics_path}"
            )

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        hp = metrics.get('hyperparameters', {})
        net_config = metrics.get('network_config', {})

        # Extract required hyperparameters (with defaults)
        belief_latent_dim = hp.get('BELIEF_LATENT_DIM', 16)
        batch_size = hp.get('BATCH_SIZE', 128)
        lr_actor = hp.get('LR_ACTOR', 0.0001)
        lr_critic = hp.get('LR_CRITIC', 0.001)
        learn_step = hp.get('LEARN_STEP', 100)
        gamma = hp.get('GAMMA', 0.95)

        # Get observation/action spaces
        observation_spaces = [env.observation_space(a) for a in agent_ids]
        action_spaces = [env.action_space(a) for a in agent_ids]

        print(f"Loading SMPE with belief_latent_dim={belief_latent_dim}")

        # Create SMPE instance
        model = SMPE(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
            net_config=net_config,
            batch_size=batch_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            learn_step=learn_step,
            gamma=gamma,
            belief_latent_dim=belief_latent_dim,
            recon_coef=1.0,  # defaults
            kl_coef=0.001,
            filter_reg_coef=0.001,
            intrinsic_coef=0.0,
            device=device,
        )

        # CRITICAL: Initialize filter optimizers before loading checkpoint
        model._init_filter_optimizers()

        # Load checkpoint
        try:
            model.load_checkpoint(model_path)
            print(f"Loaded SMPE from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SMPE checkpoint: {e}")

        return model

    @staticmethod
    def prepare_for_inference(model):
        """Set SMPE to inference mode (greedy actions)."""
        model.set_training_mode(False)
        return model


def get_model_loader(algorithm: str) -> ModelLoader:
    """
    Get appropriate model loader for algorithm.

    Args:
        algorithm: Algorithm name ("MATD3" or "SMPE")

    Returns:
        ModelLoader subclass for the algorithm

    Raises:
        ValueError: If algorithm is not supported
    """
    loaders = {
        'MATD3': MATD3Loader,
        'SMPE': SMPELoader,
    }

    if algorithm not in loaders:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Supported: {list(loaders.keys())}"
        )

    return loaders[algorithm]


def main():
    parser = argparse.ArgumentParser(description='Replay multi-agent RL models on simple_speaker_listener')
    parser.add_argument('--model', type=str, help='Experiment ID or model path. If not provided, uses latest from experiments.csv')
    parser.add_argument('--output', type=str, help='Output gif path', default=None)
    parser.add_argument('--algorithm', type=str, choices=['MATD3', 'SMPE'],
                       help='Force specific algorithm (auto-detected if not provided)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine experiment ID
    if args.model:
        exp_id = args.model
    else:
        # Get latest experiment ID
        exp_id = get_latest_experiment()
        if exp_id is None:
            raise ValueError("No model path provided and no experiments found in results/experiments.csv")

    model_path = f"results/{exp_id}/{exp_id}_model.pt"

    # Detect or use specified algorithm
    if args.algorithm:
        algorithm = args.algorithm
        print(f"Using specified algorithm: {algorithm}")
    else:
        algorithm = detect_algorithm(exp_id)
        print(f"Auto-detected algorithm: {algorithm}")

    # Create environment configured for this algorithm
    env = create_environment(algorithm, render_mode="rgb_array")
    env.reset()

    # Get agent information
    n_agents = env.num_agents
    agent_ids = env.agents

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"results/{exp_id}/{exp_id}_speaker_listener.gif"

    # Load model using appropriate loader
    loader = get_model_loader(algorithm)
    model = loader.load(model_path, env, agent_ids, device)
    model = loader.prepare_for_inference(model)

    # Define test loop parameters
    episodes = 15  # Number of episodes to test agent on
    max_steps = 25  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    # Test loop for inference
    for ep in range(episodes):
        obs, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            # Get next action from agent
            action, raw_action = model.get_action(obs, infos=info)
            # Debug prints: shapes, dtypes and sample values to diagnose listener behaviour
            try:
                for aid in raw_action:
                    a = raw_action[aid]
                    if isinstance(a, np.ndarray):
                        print(f"[DEBUG] raw_action[{aid}] shape={a.shape} dtype={a.dtype} min={a.min():.3f} max={a.max():.3f}")
                    else:
                        print(f"[DEBUG] raw_action[{aid}] type={type(a)} repr={repr(a)[:200]}")
            except Exception as e:
                print(f"[DEBUG] failed to print raw_action info: {e}")

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            # Ensure actions are in the format env.step expects (arrays or scalars per agent)
            env_action = {}
            try:
                import gym
            except Exception:
                gym = None

            for agent, a in action.items():
                # If numpy array of shape (1, ...), squeeze to remove batch dim
                if isinstance(a, np.ndarray) and a.shape and a.shape[0] == 1:
                    a = a.squeeze(0)

                # Print processed action for diagnosis
                try:
                    print(f"[DEBUG] processed_action[{agent}] type={type(a)} repr={getattr(a, '__repr__', lambda: str(a))()[:200]}")
                except Exception:
                    pass

                # Convert action to match env space expectations.
                try:
                    # Try several ways to get the agent's action space from the env
                    space = None
                    for attr in ('single_action_space', 'action_space', 'action_space_for', 'action_spaces'):
                        try:
                            accessor = getattr(env, attr)
                        except Exception:
                            continue
                        try:
                            if callable(accessor):
                                space = accessor(agent)
                            else:
                                # Could be a dict-like mapping
                                if isinstance(accessor, dict):
                                    space = accessor.get(agent, None)
                                else:
                                    space = accessor
                        except Exception:
                            # ignore and try next
                            space = None
                        if space is not None:
                            break

                    # If action is an ndarray, ensure it matches Box shape and scale if needed
                    if isinstance(a, np.ndarray):
                        a = np.asarray(a)
                        # remove leading batch dim if present
                        if a.ndim > 1 and a.shape[0] == 1:
                            a = a.squeeze(0)

                        # Check if this is a discrete action space
                        from gymnasium import spaces as gym_spaces
                        if space is not None and isinstance(space, (gym_spaces.Discrete, type(space).__name__ == 'Discrete')):
                            # For discrete actions, convert to Python int
                            a = int(a.item()) if a.size == 1 else int(a)
                        elif space is not None and hasattr(space, 'low') and hasattr(space, 'high'):
                            low = np.array(space.low, dtype=np.float32)
                            high = np.array(space.high, dtype=np.float32)
                            # Flatten and reshape if sizes mismatch
                            needed = int(np.prod(low.shape))
                            flat = a.flatten()
                            if flat.size < needed:
                                # pad with zeros if too short
                                flat = np.pad(flat, (0, needed - flat.size), mode='constant')
                            flat = flat[:needed]
                            a = flat.reshape(low.shape)

                            # If actor outputs are in [0,1], rescale to [low, high]
                            amin, amax = float(a.min()), float(a.max())
                            if amin >= 0.0 and amax <= 1.0:
                                a = low + a * (high - low)
                        else:
                            # leave as float array
                            a = a.astype(np.float32)

                    else:
                        # scalar-like actions: coerce numpy scalars to python types
                        if isinstance(a, (np.integer, np.floating)):
                            a = a.item()
                except Exception as e:
                    print(f"[DEBUG] failed to convert action for {agent}: {e}")

                env_action[agent] = a

            obs, reward, termination, truncation, info = env.step(env_action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the gif to specified path
    imageio.mimwrite(output_path, frames, duration=10)
    print(f"Saved gif to: {output_path}")


if __name__ == "__main__":
    main()