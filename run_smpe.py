"""This tutorial shows how to train an SMPE agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
Adapted to SMPE.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpe2 import simple_speaker_listener_v4

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from smpe import SMPE
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo (SMPE) =====")

    # Define the network configuration
    # NOTE: SMPE currently does not use NET_CONFIG internally,
    # but create_population still requires it.

    NET_CONFIG = {
        "latent_dim": 64,
        "encoder_config": {
            "hidden_size": [64],
        },
        "head_config": {
            "hidden_size": [64],
        },
    }
    print(f"NET_CONFIG assigned: {NET_CONFIG}")

    # Define the initial hyperparameters for SMPE
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": "SMPE",        # Algorithm name (must match registry)
        "BATCH_SIZE": 128,     # Batch size
        "LR_ACTOR": 0.0001,    # Actor learning rate
        "LR_CRITIC": 0.001,    # Critic learning rate
        "GAMMA": 0.95,         # Discount factor
        "MEMORY_SIZE": 100000, # Max memory buffer size
        "LEARN_STEP": 100,     # Learning frequency
        "TAU": 0.01,           # Unused by SMPE but kept for consistency
        # SMPE-specific (if your SMPE.__init__ reads it from net_config you can omit this):
        "BELIEF_LATENT_DIM": 16,  # Will be mapped to belief_latent_dim if supported
    }
    print(f"INIT_HP assigned: {INIT_HP}")

    num_envs = 8
    print(f"num_envs assigned: {num_envs}")

    def make_env():
        # SMPE supports *discrete* actions; use continuous_actions=False
        return simple_speaker_listener_v4.parallel_env(continuous_actions=False)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)
    print(f"env created with agents: {getattr(env, 'agents', None)}")

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    print(f"observation_spaces assigned: {[str(s) for s in observation_spaces]}")

    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    print(f"action_spaces assigned: {[str(s) for s in action_spaces]}")

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents
    print(f"INIT_HP['AGENT_IDS'] assigned: {INIT_HP['AGENT_IDS']}")

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )
    print(f"hp_config assigned: {hp_config}")

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop: list[SMPE] = []
    for idx in range(INIT_HP["POPULATION_SIZE"]):
        agent = SMPE(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=INIT_HP["AGENT_IDS"],
            batch_size=INIT_HP["BATCH_SIZE"],
            lr_actor=INIT_HP["LR_ACTOR"],
            lr_critic=INIT_HP["LR_CRITIC"],
            learn_step=INIT_HP["LEARN_STEP"],
            gamma=INIT_HP["GAMMA"],
            belief_latent_dim=INIT_HP["BELIEF_LATENT_DIM"],
            hp_config=hp_config,
            index=idx,        # important for tournament/mutation bookkeeping
            device=device,
        )
        pop.append(agent)

    print(f"pop created with size: {len(pop)}")

    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    print(f"field_names assigned: {field_names}")

    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )
    print(f"memory created: size={INIT_HP['MEMORY_SIZE']} agents={INIT_HP['AGENT_IDS']}")

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,       # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,        # Evaluate using last N fitness scores
    )
    print(f"tournament assigned: {tournament}")

    # Instantiate a mutations object (used for HPO)
    # For SMPE, disable architecture/parameter mutations, keep only RL HP mutations.
    mutations = Mutations(
        no_mutation=0.2,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.0,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=1,
        device=device,
    )
    print(f"mutations assigned: {mutations}")

    # Define training loop parameters
    max_steps = 2_000_000  # Max steps (default: 2000000)
    print(f"max_steps assigned: {max_steps}")
    learning_delay = 0     # Steps before starting learning
    print(f"learning_delay assigned: {learning_delay}")
    evo_steps = 1_000     # Evolution frequency
    print(f"evo_steps assigned: {evo_steps}")
    eval_steps = None      # Evaluation steps per episode - go until done
    print(f"eval_steps assigned: {eval_steps}")
    eval_loop = 1          # Number of evaluation episodes
    print(f"eval_loop assigned: {eval_loop}")
    elite = pop[0]         # Assign a placeholder "elite" agent
    print(f"elite assigned: {elite}")
    total_steps = 0
    print(f"total_steps assigned: {total_steps}")
    
    # Lista para armazenar pontuações médias para plotagem
    training_scores_history = []
    print("training_scores_history assigned: []")

    # TRAINING LOOP
    print("Training...")
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
                # For SMPE, both returned dicts contain discrete int actions
                action, _ = agent.get_action(
                    obs=obs, infos=info
                )  # Predict action
                next_obs, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Save experiences to replay buffer
                # IMPORTANT: For SMPE we store the same actions we pass to the env
                memory.save_to_memory(
                    obs,
                    action,
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

                # Calculate scores and handle finished episodes
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

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

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Create timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./models/SMPE/{timestamp}"
    os.makedirs(path, exist_ok=True)

    # Save the trained algorithm
    filename = "SMPE_trained_agent.pt"
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)
    print(f"Modelo salvo em: {save_path}")

    # Save hyperparameters to txt file
    params_path = os.path.join(path, "hyperparameters.txt")
    with open(params_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Training Run: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("INITIAL HYPERPARAMETERS:\n")
        f.write("-" * 60 + "\n")
        for key, value in INIT_HP.items():
            if key != "AGENT_IDS":  # Skip agent IDs (they're env-specific)
                f.write(f"{key:20s} = {value}\n")

        f.write("\n")
        f.write("NETWORK CONFIGURATION:\n")
        f.write("-" * 60 + "\n")
        for key, value in NET_CONFIG.items():
            f.write(f"{key:20s} = {value}\n")

        f.write("\n")
        f.write("TRAINING SETTINGS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'max_steps':20s} = {max_steps}\n")
        f.write(f"{'num_envs':20s} = {num_envs}\n")
        f.write(f"{'evo_steps':20s} = {evo_steps}\n")
        f.write(f"{'device':20s} = {device}\n")

        f.write("\n")
        f.write("FINAL RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Final Mean Score':20s} = {training_scores_history[-1]:.2f}\n")
        f.write(f"{'Best Score':20s} = {max(training_scores_history):.2f}\n")
        f.write(f"{'Worst Score':20s} = {min(training_scores_history):.2f}\n")

    print(f"Hiperparâmetros salvos em: {params_path}")

    # Create symlink to latest model for easy access
    latest_link = "./models/SMPE/latest"
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(timestamp, latest_link, target_is_directory=True)
    print(f"Link simbólico 'latest' criado apontando para: {timestamp}")

    # Plotar e salvar a evolução das pontuações
    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Evolução das Pontuações Médias Durante o Treinamento (SMPE)', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média da População', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Salvar o gráfico
    plot_path = os.path.join(path, "training_scores_evolution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de evolução das pontuações salvo em: {plot_path}")

    # Salvar dados das pontuações em arquivo numpy
    scores_data_path = os.path.join(path, "training_scores_history.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    print(f"Dados das pontuações salvos em: {scores_data_path}")

    plt.show()

    pbar.close()
    env.close()
