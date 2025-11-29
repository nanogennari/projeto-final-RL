import os
import argparse
import pandas as pd
import imageio
import numpy as np
import torch
from mpe2 import simple_speaker_listener_v4
from PIL import Image, ImageDraw

from agilerl.algorithms import MATD3


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


def main():
    parser = argparse.ArgumentParser(description='Test MATD3 on simple_speaker_listener')
    parser.add_argument('--model', type=str, help='Path to the model file. If not provided, uses latest from experiments.csv')
    parser.add_argument('--output', type=str, help='Output gif path', default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_speaker_listener_v4.parallel_env(
        continuous_actions=True, render_mode="rgb_array"
    )
    env.reset()

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Determine model path
    if args.model:
        exp_id = args.model
    else:
        # Get latest experiment ID
        exp_id = get_latest_experiment()
        if exp_id is None:
            raise ValueError("No model path provided and no experiments found in results/experiments.csv")
    model_path = f"results/{exp_id}/{exp_id}_model.pt"

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.model:
            # Extract exp_name from model path
            exp_name = os.path.splitext(os.path.basename(args.model))[0].replace('_model', '')
        else:
            exp_name = get_latest_experiment()
        output_path = f"results/{exp_name}/{exp_name}_speaker_listener.gif"

    # Load the saved agent
    matd3 = MATD3.load(model_path, device)
    print(f"Loaded model from: {model_path}")

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
            action, _ = matd3.get_action(obs, infos=info)

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            obs, reward, termination, truncation, info = env.step(
                {agent: a.squeeze() for agent, a in action.items()}
            )

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