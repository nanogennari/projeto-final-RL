"""
MATD3 Multi-Agent Training with Experiment Management

Features:
- Live progress monitoring
- YAML-based configuration
- Results tracking and comparison

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
Modified for experiment management
"""

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpe2 import simple_speaker_listener_v4

from agilerl.algorithms import MATD3
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

# Import experiment management modules
from src.experiment_manager import ExperimentManager
from src.progress_tracker import ProgressTracker
from src.results_tracker import ResultsTracker

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MATD3 Multi-Agent Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/baseline.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Experiment ID to resume (defaults to auto-detect from config)",
    )
    args = parser.parse_args()

    # Load experiment configuration
    print("=" * 60)
    print("MATD3 Multi-Agent Training with Experiment Management")
    print("=" * 60)
    print()
    print(f"Loading config: {args.config}")

    exp_manager = ExperimentManager(args.config)
    config = exp_manager.load_config()

    print(f"Experiment: {config.name}")
    print(f"Description: {config.description}")
    print()

    # Set random seeds for reproducibility
    set_seeds(config.seed)
    print(f"✓ Random seed set: {config.seed}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    print()

    # Load configurations from YAML
    INIT_HP = exp_manager.get_init_hp()
    NET_CONFIG = exp_manager.get_net_config()
    hp_config = exp_manager.get_hpo_config()
    mutation_config = exp_manager.get_mutation_config()
    training_config = exp_manager.get_training_config()

    # Training parameters
    max_steps = training_config["max_steps"]
    num_envs = training_config["num_envs"]
    evo_steps = training_config["evo_steps"]
    learning_delay = training_config["learning_delay"]
    eval_steps = training_config["eval_steps"]
    eval_loop = training_config["eval_loop"]

    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Create experiment ID for this run
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize experiment management
    print("-" * 60)
    print("Initializing experiment management...")
    print("-" * 60)

    progress_tracker = ProgressTracker(
        experiment_id=exp_id,
        progress_dir="progress",
        max_steps=max_steps,
    )

    results_tracker = ResultsTracker(results_dir="results")

    print(f"✓ Experiment ID: {exp_id}")
    print(f"✓ Progress tracking enabled")
    print()

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop: list[MATD3] = create_population(
        INIT_HP["ALGO"],
        NET_CONFIG,
        INIT_HP,
        observation_spaces,
        action_spaces,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer (only for off-policy algorithms)
    # IPPO is on-policy and doesn't use replay buffer
    if INIT_HP["ALGO"] != "IPPO":
        field_names = ["obs", "action", "reward", "next_obs", "done"]
        memory = MultiAgentReplayBuffer(
            INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            agent_ids=INIT_HP["AGENT_IDS"],
            device=device,
        )
    else:
        memory = None  # IPPO doesn't use replay buffer

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO) - from config
    mutations = Mutations(
        no_mutation=mutation_config["no_mutation"],
        architecture=mutation_config["architecture"],
        new_layer_prob=mutation_config["new_layer_prob"],
        parameters=mutation_config["parameters"],
        activation=mutation_config["activation"],
        rl_hp=mutation_config["rl_hp"],
        mutation_sd=mutation_config["mutation_sd"],
        rand_seed=config.seed,
        device=device,
    )

    # Initialize training variables
    total_steps = 0
    training_scores_history = []

    # Register new experiment
    results_tracker.register_experiment(
        exp_id=exp_id,
        name=config.name,
        config_path=args.config,
        start_time=datetime.now(),
    )

    print("-" * 60)
    print("Starting training...")
    print("-" * 60)
    print()

    # Initialize training variables
    elite = pop[0]  # Placeholder elite agent
    start_time = time.time()

    # TRAINING LOOP
    print("-" * 60)
    print(f"Starting training: {total_steps:,} / {max_steps:,} steps")
    print("-" * 60)
    print()
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.set_training_mode(True)
            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            for idx_step in range(evo_steps // num_envs):
                action, raw_action = agent.get_action(
                    obs=obs, infos=info
                )  # Predict action
                next_obs, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Save experiences to replay buffer
                memory.save_to_memory(
                    obs,
                    raw_action,
                    reward,
                    next_obs,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else 0
            )
            for episode_scores in pop_episode_scores
        ]
        
        # Salvar pontuação média da população para plotagem
        population_mean_score = np.mean([score for score in mean_scores if isinstance(score, (int, float))])
        training_scores_history.append(population_mean_score)

        mean_scores_display = [
            (
                score if isinstance(score, (int, float))
                else "0 completed episodes"
            )
            for score in mean_scores
        ]

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps {[agent.steps[-1] for agent in pop]}\n"
            f"Scores: {mean_scores_display}\n"
            f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
            f"Mutations: {[agent.mut for agent in pop]}"
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update progress tracker
        elapsed_time = time.time() - start_time
        fitness_avg = np.mean([agent.fitness[-5:] for agent in pop if len(agent.fitness) >= 5])
        best_score = max([s for s in mean_scores if isinstance(s, (int, float))], default=0)
        worst_score = min([s for s in mean_scores if isinstance(s, (int, float))], default=0)

        progress_tracker.update(
            step=total_steps,
            elapsed_time=elapsed_time,
            mean_score=population_mean_score,
            best_score=best_score,
            worst_score=worst_score,
            fitness_avg=fitness_avg,
            elite_mutations=elite.mut if hasattr(elite, 'mut') else 0,
        )

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Training complete!
    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()

    # Calculate final metrics
    final_mean_score = training_scores_history[-1] if training_scores_history else 0
    best_score = max(training_scores_history) if training_scores_history else 0
    worst_score = min(training_scores_history) if training_scores_history else 0
    total_duration_hours = (time.time() - start_time) / 3600

    # Save model temporarily
    temp_model_path = f"/tmp/{exp_id}_model.pt"
    elite.save_checkpoint(temp_model_path)

    # Create plots
    temp_plot_path = f"/tmp/{exp_id}_plot.png"
    temp_data_path = f"/tmp/{exp_id}_data.npy"

    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Evolução das Pontuações Médias Durante o Treinamento', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média da População', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(temp_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    np.save(temp_data_path, np.array(training_scores_history))

    # Save results using results tracker
    results_tracker.save_training_artifacts(
        exp_id=exp_id,
        model_path=temp_model_path,
        scores_plot_path=temp_plot_path,
        scores_data_path=temp_data_path,
    )

    # Save detailed metrics JSON
    metrics = {
        "experiment": {
            "id": exp_id,
            "name": config.name,
            "config": args.config,
            "seed": config.seed,
        },
        "training": {
            "steps": total_steps,
            "duration_hours": total_duration_hours,
            "num_envs": num_envs,
            "evo_steps": evo_steps,
        },
        "final_performance": {
            "mean_score": float(final_mean_score),
            "best_score": float(best_score),
            "worst_score": float(worst_score),
            "std_score": float(np.std(training_scores_history)) if training_scores_history else 0,
        },
        "elite_agent": {
            "lr_actor": elite.lr_actor if hasattr(elite, 'lr_actor') else INIT_HP["LR_ACTOR"],
            "lr_critic": elite.lr_critic if hasattr(elite, 'lr_critic') else INIT_HP["LR_CRITIC"],
            "batch_size": elite.batch_size if hasattr(elite, 'batch_size') else INIT_HP["BATCH_SIZE"],
            "mutations_applied": elite.mut if hasattr(elite, 'mut') else 0,
        },
    }

    results_tracker.save_metrics_json(exp_id, metrics)

    # Finalize experiment in registry
    results_tracker.finalize_experiment(
        exp_id=exp_id,
        steps=total_steps,
        duration_hours=total_duration_hours,
        final_score=final_mean_score,
        best_score=best_score,
        worst_score=worst_score,
    )

    # Summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Experiment ID:     {exp_id}")
    print(f"Total Steps:       {total_steps:,}")
    print(f"Duration:          {total_duration_hours:.2f} hours")
    print(f"Final Mean Score:  {final_mean_score:.2f}")
    print(f"Best Score:        {best_score:.2f}")
    print(f"Worst Score:       {worst_score:.2f}")
    print()
    print(f"Results saved in:  results/{exp_id}/")
    print(f"Registry updated:  results/experiments.csv")
    print()
    print("=" * 60)
    print()
    print("TIP: Compare experiments with: python compare.py <exp_id1> <exp_id2>")
    print()

    pbar.close()
    env.close()
