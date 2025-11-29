"""This tutorial shows how to train an SMPE agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
Adapted to SMPE.
"""
import os
from datetime import datetime
import math
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from mpe2 import simple_speaker_listener_v4

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter, NetworkGroup
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.base import EvolvableModule, ModuleDict
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.configs import MlpNetConfig
from agilerl.typing import (
    ArrayDict,
    InfosDict,
    MultiAgentModule,
    ObservationType,
    PzEnvType,
    StandardTensorDict,
)
from agilerl.utils.utils import default_progress_bar, make_multi_agent_vect_envs
from agilerl.utils.algo_utils import key_in_nested_dict, obs_channels_to_first
import json
import csv
import shutil
from pathlib import Path


# ==========================
#   Non-evolvable helpers
# ==========================


# class SMPEFilter(nn.Module):
#     """Agent Modelling filter – learnable feature-wise gating on other agents' info."""

#     def __init__(self, filter_dim: int, device: str = 'cpu'):
#         super().__init__()
#         self.filter_dim = filter_dim
#         self.filter_params = nn.Parameter(torch.ones(filter_dim, device=device))

#     def forward(self) -> torch.Tensor:
#         # Constrained to [0, 1]
#         return torch.sigmoid(self.filter_params)

# AMFilter doesn't need to be evolvable - it's just a parameter vector optimized via gradients
class AMFilter(nn.Module):
    def __init__(
        self,
        filter_dim: int,
        device: str = "cpu",
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        self.filter_dim = filter_dim
        self.device = device
        self.random_seed = random_seed

        # Actual parameters
        self.filter_params = nn.Parameter(torch.ones(filter_dim, device=device))

    def forward(self) -> torch.Tensor:
        """Return per-feature gate in [0, 1]."""
        return torch.sigmoid(self.filter_params)


class BeliefNoveltyModule:
    """Simple SimHash-based intrinsic reward from belief novelty."""

    def __init__(self, belief_dim: int, n_bits: int = 32):
        self.proj_matrix = torch.randn(belief_dim, n_bits)
        self.proj_matrix.requires_grad = False
        self.counts: dict[int, int] = {}

    def _hash_belief(self, belief_vec: torch.Tensor) -> int:
        v = belief_vec.detach().cpu()
        bits = (v @ self.proj_matrix >= 0).numpy().astype("int64")
        h = 0
        for bit in bits[::-1]:
            h = (h << 1) | int(bit)
        return int(h)

    def get_intrinsic_reward(self, belief_vec: torch.Tensor) -> float:
        key = self._hash_belief(belief_vec)
        prev = self.counts.get(key, 0)
        self.counts[key] = prev + 1
        return 1.0 / math.sqrt(self.counts[key])


# ==========================
#          SMPE
# ==========================


class SMPE(MultiAgentRLAlgorithm):
    """SMPE algorithm adapted to AgileRL with evolvable architectures.

    - Discrete actions.
    - Per-agent encoder / decoder / actor / critics implemented with EvolvableMLP.
      => fully compatible with architecture mutation via Mutations(architecture > 0).

    This matches the MATD3-like interface used in your training script.
    """

    possible_action_spaces: dict[str, Union[spaces.Discrete, spaces.Box]]

    # Evolvable networks
    encoders: MultiAgentModule[EvolvableMLP]
    decoders: MultiAgentModule[EvolvableMLP]
    actors: MultiAgentModule[EvolvableMLP]
    critics_main: MultiAgentModule[EvolvableMLP]
    critics_filtered: MultiAgentModule[EvolvableMLP]

    def __init__(
        self,
        observation_spaces: Union[list[spaces.Space], spaces.Dict],
        action_spaces: Union[list[spaces.Space], spaces.Dict],
        agent_ids: Optional[list[str]] = None,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[dict[str, Any]] = None,
        batch_size: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        learn_step: int = 100,
        gamma: float = 0.95,
        # SMPE-specific
        belief_latent_dim: int = 16,
        recon_coef: float = 1.0,
        kl_coef: float = 1e-3,
        filter_reg_coef: float = 1e-3,
        intrinsic_coef: float = 0.0,  # set >0 to actually use intrinsic rewards
        normalize_images: bool = True,
        mut: Optional[str] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        torch_compiler: Optional[str] = None,
        wrap: bool = True,
    ):
        super().__init__(
            observation_spaces,
            action_spaces,
            index=index,
            agent_ids=agent_ids,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            torch_compiler=torch_compiler,
            name="SMPE",
        )

        # ------------- sanity checks / basic attrs -------------
        assert isinstance(batch_size, int) and batch_size >= 1
        assert isinstance(lr_actor, float) and lr_actor > 0.0
        assert isinstance(lr_critic, float) and lr_critic > 0.0
        assert isinstance(learn_step, int) and learn_step >= 1
        assert isinstance(gamma, float)
        assert isinstance(belief_latent_dim, int) and belief_latent_dim > 0
        assert isinstance(wrap, bool)

        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learn_step = learn_step
        self.gamma = gamma
        self.mut = mut
        self.recon_coef = recon_coef
        self.kl_coef = kl_coef
        self.filter_reg_coef = filter_reg_coef
        self.intrinsic_coef = intrinsic_coef
        self.net_config = net_config or {}
        self.belief_latent_dim = belief_latent_dim

        # SMPE: no explicit action noise, but training loop expects this method
        self.O_U_noise = False

        # --------- per-agent dimensions (discrete actions assumed) ----------
        self.obs_dims: dict[str, int] = {}
        self.action_dims: dict[str, int] = {}
        for agent_id in self.agent_ids:
            obs_space = self.possible_observation_spaces[agent_id]
            act_space = self.possible_action_spaces[agent_id]
            assert isinstance(
                act_space, spaces.Discrete
            ), "Current SMPE implementation assumes discrete actions."
            assert isinstance(obs_space, spaces.Box), "Assumes vector observations."

            self.obs_dims[agent_id] = int(np.prod(obs_space.shape))
            self.action_dims[agent_id] = act_space.n

        # others obs / actions dims per agent
        self.others_obs_dims: dict[str, int] = {}
        self.others_act_dims: dict[str, int] = {}
        for agent_id in self.agent_ids:
            self.others_obs_dims[agent_id] = sum(
                self.obs_dims[a] for a in self.agent_ids if a != agent_id
            )
            self.others_act_dims[agent_id] = sum(
                self.action_dims[a] for a in self.agent_ids if a != agent_id
            )

        # ---------- net_config -> simple MLP configs ----------
        # We'll interpret your NET_CONFIG as:
        # encoder_config  -> encoder MLP
        # decoder_config  -> decoder MLP (defaults to encoder_config)
        # actor_config    -> actor head (defaults to head_config / encoder_config)
        # critic_config   -> critic head (defaults to head_config / encoder_config)
        default_encoder_cfg = asdict(MlpNetConfig(hidden_size=[64]))
        default_encoder_cfg.pop("output_activation", None)
        default_head_cfg = asdict(MlpNetConfig(hidden_size=[64]))
        default_head_cfg.pop("output_activation", None)

        encoder_cfg = (
            self.net_config.get("encoder_config", {}) or default_encoder_cfg
        )
        decoder_cfg = (
            self.net_config.get("decoder_config", {}) or encoder_cfg
        )
        actor_cfg = (
            self.net_config.get("actor_config", {})
            or self.net_config.get("head_config", {})
            or default_head_cfg
        )
        critic_cfg = (
            self.net_config.get("critic_config", {})
            or self.net_config.get("head_config", {})
            or default_head_cfg
        )

        # small helper to pick activation
        def _act(cfg: dict, default: str = "ReLU") -> str:
            return cfg.get("activation", default)

        # --------- build evolvable networks per agent ----------
        self.encoders = ModuleDict()
        self.decoders = ModuleDict()
        self.actors = ModuleDict()
        self.critics_main = ModuleDict()
        self.critics_filtered = ModuleDict()
        # Use underscore prefix to exclude from AgileRL's evolvable registry
        self._filters = ModuleDict(
                    {       agent_id: AMFilter(
                            filter_dim=self.others_obs_dims[agent_id] + self.others_act_dims[agent_id], device = self.device)
                        for agent_id in self.agent_ids})
        self.novelty_modules: dict[str, BeliefNoveltyModule] = {}

        for agent_id in self.agent_ids:
            obs_dim = self.obs_dims[agent_id]
            act_dim = self.action_dims[agent_id]
            others_obs_dim = self.others_obs_dims[agent_id]
            others_act_dim = self.others_act_dims[agent_id]

            # Encoder: [obs, prev_action_one_hot] -> [mu || logvar]
            encoder = EvolvableMLP(
                num_inputs=obs_dim + act_dim,
                num_outputs=2 * belief_latent_dim,
                hidden_size=encoder_cfg.get("hidden_size", [64]),
                activation=_act(encoder_cfg),
                output_activation=None,
            )

            # Decoder: belief -> reconstruct [others_obs, others_act]
            decoder = EvolvableMLP(
                num_inputs=belief_latent_dim,
                num_outputs=others_obs_dim + others_act_dim,
                hidden_size=decoder_cfg.get("hidden_size", [64]),
                activation=_act(decoder_cfg),
                output_activation=None,
            )

            # Actor: [own_obs, belief] -> action logits
            actor = EvolvableMLP(
                num_inputs=obs_dim + belief_latent_dim,
                num_outputs=act_dim,
                hidden_size=actor_cfg.get("hidden_size", [64]),
                activation=_act(actor_cfg),
                output_activation=None,  # we treat outputs as logits
            )

            # Critics: V(full_state) and V(filtered_state)
            state_dim = obs_dim + others_obs_dim
            critic_main = EvolvableMLP(
                num_inputs=state_dim,
                num_outputs=1,
                hidden_size=critic_cfg.get("hidden_size", [64]),
                activation=_act(critic_cfg),
                output_activation=None,
            )
            critic_filt = EvolvableMLP(
                num_inputs=state_dim,
                num_outputs=1,
                hidden_size=critic_cfg.get("hidden_size", [64]),
                activation=_act(critic_cfg),
                output_activation=None,
            )

            # Filter + novelty module
            #filt = AMFilter(filter_dim=others_obs_dim + others_act_dim)
            novelty = BeliefNoveltyModule(belief_dim=belief_latent_dim)

            self.encoders[agent_id] = encoder.to(self.device)
            self.decoders[agent_id] = decoder.to(self.device)
            self.actors[agent_id] = actor.to(self.device)
            self.critics_main[agent_id] = critic_main.to(self.device)
            self.critics_filtered[agent_id] = critic_filt.to(self.device)
            # Filters are already in ModuleDict created with correct device at line 255-258, no need to move
            # self.filters[agent_id] = self.filters[agent_id].to(self.device)
            self.novelty_modules[agent_id] = novelty

        # Per-agent prev actions (for belief input) stored as one-hot
        self.prev_actions: dict[str, torch.Tensor] = {}

        # --------- optimizers ----------
        self.actor_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.actors, lr=lr_actor
        )
        self.encoder_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.encoders, lr=lr_critic
        )
        self.decoder_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.decoders, lr=lr_critic
        )
        # Filter optimizers will be created after registry init to avoid detection
        # Initialized in _init_filter_optimizers() called after __init__
        self.critic_main_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.critics_main, lr=lr_critic
        )
        self.critic_filt_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.critics_filtered, lr=lr_critic
        )

        # no special torch.compile logic here; SMPE is light & discrete

        self.value_criterion = nn.MSELoss()

        # --------- register network groups for mutation ----------
        # Actor = policy group
        self.register_network_group(
            NetworkGroup(eval_network=self.actors, policy=True)
        )
        # Critic groups
        self.register_network_group(
            NetworkGroup(eval_network=self.critics_main)
        )
        self.register_network_group(
            NetworkGroup(eval_network=self.critics_filtered)
        )
        # Generative model groups (encoder / decoder)
        self.register_network_group(
            NetworkGroup(eval_network=self.encoders)
        )
        self.register_network_group(
            NetworkGroup(eval_network=self.decoders)
        )
        # Filters are not evolvable, so don't register them as a network group

    def _init_filter_optimizers(self) -> None:
        """Initialize filter optimizers after registry init to avoid detection."""
        self._filter_optimizers = {
            agent_id: optim.Adam(self._filters[agent_id].parameters(), lr=self.lr_critic)
            for agent_id in self.agent_ids
        }

    def clone(self, index=None, wrap=True):
        """Override clone to handle non-evolvable filter optimizers."""
        # Temporarily remove filter optimizers to avoid registry detection
        filter_opts = getattr(self, '_filter_optimizers', None)
        if filter_opts is not None:
            del self._filter_optimizers

        # Call parent clone
        clone = super().clone(index=index, wrap=wrap)

        # Restore filter optimizers on original
        if filter_opts is not None:
            self._filter_optimizers = filter_opts

        # Clone will get its optimizers in mutation_hook
        return clone

    def save_checkpoint(self, path: str) -> None:
        """Override save_checkpoint to handle non-evolvable filter optimizers."""
        # Temporarily remove filter optimizers to avoid registry detection
        filter_opts = getattr(self, '_filter_optimizers', None)
        if filter_opts is not None:
            del self._filter_optimizers

        # Call parent save_checkpoint
        super().save_checkpoint(path)

        # Restore filter optimizers
        if filter_opts is not None:
            self._filter_optimizers = filter_opts

    def mutation_hook(self) -> None:
        """Ensure all modules stay on the correct device after mutations/cloning."""
        # Move all evolvable modules to the correct device
        for agent_id in self.agent_ids:
            if agent_id in self.encoders:
                self.encoders[agent_id] = self.encoders[agent_id].to(self.device)
            if agent_id in self.decoders:
                self.decoders[agent_id] = self.decoders[agent_id].to(self.device)
            if agent_id in self.actors:
                self.actors[agent_id] = self.actors[agent_id].to(self.device)
            if agent_id in self.critics_main:
                self.critics_main[agent_id] = self.critics_main[agent_id].to(self.device)
            if agent_id in self.critics_filtered:
                self.critics_filtered[agent_id] = self.critics_filtered[agent_id].to(self.device)

        # Filters need manual cloning since they're not EvolvableModules
        # Clone filters from the original instance
        for agent_id in self.agent_ids:
            if agent_id in self._filters:
                # Create a new filter with the same parameters
                old_filter = self._filters[agent_id]
                new_filter = AMFilter(
                    filter_dim=old_filter.filter_dim,
                    device=self.device,
                    random_seed=old_filter.random_seed
                ).to(self.device)
                # Copy the learned parameters
                new_filter.load_state_dict(old_filter.state_dict())
                self._filters[agent_id] = new_filter

        # Recreate filter optimizers with new filter instances
        # This also handles initialization for cloned instances
        self._filter_optimizers = {
            agent_id: optim.Adam(self._filters[agent_id].parameters(), lr=self.lr_critic)
            for agent_id in self.agent_ids
        }

        # Call parent mutation_hook to execute any registered hooks
        super().mutation_hook()

    # ==========================================================
    #   Helper: infos / masks (copied from MATD3 style)
    # ==========================================================

    def process_infos(
        self, infos: Optional[InfosDict] = None
    ) -> tuple[ArrayDict, ArrayDict, ArrayDict]:
        """Extract masks and env-defined actions from info dicts."""
        if infos is None:
            infos = {agent: {} for agent in self.agent_ids}

        env_defined_actions, agent_masks = self.extract_agent_masks(infos)
        action_masks = self.extract_action_masks(infos)
        return action_masks, env_defined_actions, agent_masks

    # ==========================================================
    #   Action selection
    # ==========================================================

    def _ensure_prev_actions(
        self, agent_id: str, batch_size: int
    ) -> torch.Tensor:
        if (
            agent_id not in self.prev_actions
            or self.prev_actions[agent_id].shape[0] != batch_size
        ):
            self.prev_actions[agent_id] = torch.zeros(
                batch_size,
                self.action_dims[agent_id],
                device=self.device,
                dtype=torch.float32,
            )
        return self.prev_actions[agent_id]

    def _encode_belief(
        self, agent_id: str, obs: torch.Tensor, prev_action_oh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder: [obs, prev_action] -> (mu, logvar, belief)."""
        enc = self.encoders[agent_id]
        enc_device = next(enc.parameters()).device
        obs = obs.to(enc_device)
        prev_action_oh = prev_action_oh.to(enc_device)

        x = torch.cat([obs, prev_action_oh], dim=-1)
        out = enc(x)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        belief = mu + eps * std
        return mu, logvar, belief

    def get_action(
        self, obs: dict[str, ObservationType], infos: Optional[InfosDict] = None
    ) -> tuple[ArrayDict, ArrayDict]:
        """Return discrete actions for each agent as in your SMPE script.

        We treat network outputs as logits of a Categorical.
        """
        assert not key_in_nested_dict(
            obs, "action_mask"
        ), "AgileRL requires action masks to be defined in the info dict."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)

        # Preprocess observations -> tensors
        preprocessed_states = self.preprocess_observation(obs)

        action_dict: dict[str, np.ndarray] = {}
        for agent_id, obs_tensor in preprocessed_states.items():
            actor = self.actors[agent_id]
            actor.eval()

            batch_size = obs_tensor.shape[0]
            prev_action_oh = self._ensure_prev_actions(agent_id, batch_size)

            # Encode current belief
            with torch.no_grad():
                _, _, belief = self._encode_belief(agent_id, obs_tensor, prev_action_oh)

                # 2) Make sure obs_tensor and belief are on the SAME device as the actor
                actor_device = next(actor.parameters()).device
                obs_tensor = obs_tensor.to(actor_device)
                belief = belief.to(actor_device)

                actor_input = torch.cat([obs_tensor, belief], dim=-1)
                logits = actor(actor_input)

                if self.training:
                    dist = torch.distributions.Categorical(logits=logits)
                    actions_idx = dist.sample()
                else:
                    actions_idx = torch.argmax(logits, dim=-1)

            # Update prev actions (one-hot)
            prev_one_hot = F.one_hot(
                actions_idx, num_classes=self.action_dims[agent_id]
            ).float()
            self.prev_actions[agent_id] = prev_one_hot

            action_dict[agent_id] = actions_idx.cpu().numpy()

            actor.train()

        # Process action masks / env-defined actions (same pattern as MATD3)
        processed_action_dict: ArrayDict = OrderedDict()
        for agent_id, action_space in self.possible_action_spaces.items():
            if isinstance(action_space, spaces.Discrete):
                action = action_dict[agent_id]
                mask = (
                    1 - np.array(action_masks[agent_id])
                    if action_masks[agent_id] is not None
                    else None
                )
                # if mask exists, just ensure we never pick masked actions.
                if mask is not None:
                    # If all masked, we keep original; else force legal via argmax over masked logits (already applied in env logic)
                    pass
                processed_action_dict[agent_id] = action
            else:
                raise RuntimeError(
                    "SMPE currently only implemented for discrete actions."
                )

        # If using env-defined_action overrides
        if env_defined_actions is not None:
            for agent in self.agent_ids:
                processed_action_dict[agent][agent_masks[agent]] = env_defined_actions[
                    agent
                ][agent_masks[agent]]

        # For SMPE, raw and processed actions are identical (discrete ints)
        return processed_action_dict, processed_action_dict

    # Training script calls this, but SMPE has no action noise.
    def reset_action_noise(self, indices: list[int]) -> None:
        return

    # ==========================================================
    #   Learning
    # ==========================================================

    def learn(self, experiences: tuple[StandardTensorDict, ...]) -> dict[str, float]:
        """Update SMPE networks from a batch of replayed experiences.

        experiences: (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences

        # Move to device
        actions = {
            a: actions[a].to(self.device) for a in self.agent_ids
        }
        rewards = {
            a: rewards[a].to(self.device) for a in self.agent_ids
        }
        dones = {
            a: dones[a].to(self.device) for a in self.agent_ids
        }

        # Preprocess obs
        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        loss_dict: dict[str, float] = {}
        for agent_id in self.agent_ids:
            total_loss = self._learn_agent(
                agent_id,
                states=states,
                next_states=next_states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            loss_dict[agent_id] = total_loss

        return loss_dict

    def _learn_agent(
        self,
        agent_id: str,
        states: StandardTensorDict,
        next_states: StandardTensorDict,
        actions: StandardTensorDict,
        rewards: StandardTensorDict,
        dones: StandardTensorDict,
    ) -> float:
        """Single-agent SMPE update, using centralized critics + auxiliary losses."""

        encoder = self.encoders[agent_id]
        decoder = self.decoders[agent_id]
        actor = self.actors[agent_id]
        critic_main = self.critics_main[agent_id]
        critic_filt = self.critics_filtered[agent_id]
        filt = self._filters[agent_id]
        novelty_mod = self.novelty_modules[agent_id]

        encoder_opt = self.encoder_optimizers[agent_id]
        decoder_opt = self.decoder_optimizers[agent_id]
        filter_opt = self._filter_optimizers[agent_id]
        actor_opt = self.actor_optimizers[agent_id]
        critic_main_opt = self.critic_main_optimizers[agent_id]
        critic_filt_opt = self.critic_filt_optimizers[agent_id]

        # Use self.device for all tensors - modules should already be on this device from __init__
        device = self.device

        # ------- Own transitions on correct device -------
        own_obs = states[agent_id].to(device)
        own_next_obs = next_states[agent_id].to(device)
        own_actions = actions[agent_id].long().view(-1).to(device)
        r = rewards[agent_id].float().view(-1).to(device)
        d = dones[agent_id].float().view(-1).to(device)

        # Replace NaNs if present (on device)
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        d = torch.where(torch.isnan(d), torch.ones_like(d), d)

        # ------- Build others_obs / others_act (current and next) -------
        others_obs_list = [
            states[a].to(device) for a in self.agent_ids if a != agent_id
        ]
        others_next_obs_list = [
            next_states[a].to(device) for a in self.agent_ids if a != agent_id
        ]
        others_act_oh_list = []
        for other_id in self.agent_ids:
            if other_id == agent_id:
                continue
            dim = self.action_dims[other_id]
            other_act = actions[other_id].long().view(-1).to(device)
            others_act_oh_list.append(
                F.one_hot(other_act, num_classes=dim).float()
            )

        if len(others_obs_list) > 0:
            others_obs = torch.cat(others_obs_list, dim=-1)               # [B, sum_obs]
            others_next_obs = torch.cat(others_next_obs_list, dim=-1)     # [B, sum_obs]
            others_act = torch.cat(others_act_oh_list, dim=-1)            # [B, sum_act]
        else:
            others_obs = torch.zeros_like(own_obs, device=device)
            others_next_obs = torch.zeros_like(own_next_obs, device=device)
            others_act = torch.zeros(
                own_obs.shape[0],
                0,
                device=device,
            )

        # ------- Critic targets -------
        full_state = torch.cat([own_obs, others_obs], dim=-1)              # [B, obs_dim + others_obs_dim]
        full_next_state = torch.cat([own_next_obs, others_next_obs], dim=-1)

        V_pred = critic_main(full_state).squeeze(-1)                       # [B]
        with torch.no_grad():
            V_next = critic_main(full_next_state).squeeze(-1)              # [B]

        # ------- Intrinsic reward from belief novelty -------
        # (we use current obs + current action as proxy for encoder's input)
        own_act_oh = F.one_hot(
            own_actions, num_classes=self.action_dims[agent_id]
        ).float().to(device)
        enc_input = torch.cat([own_obs, own_act_oh], dim=-1).to(device)
        # enc_input is explicitly on device, encoder is already on device from __init__
        enc_out = encoder(enc_input)
        mu, logvar = torch.chunk(enc_out, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        belief = mu + eps * std                                           # [B, belief_latent_dim] on device

        if self.intrinsic_coef > 0.0:
            intrinsic_r = []
            for b_vec in belief:
                intrinsic_r.append(
                    novelty_mod.get_intrinsic_reward(b_vec)
                )
            intrinsic_r = torch.tensor(
                intrinsic_r, device=device, dtype=torch.float32
            )
        else:
            intrinsic_r = torch.zeros_like(r)

        total_r = r + self.intrinsic_coef * intrinsic_r

        target_return = total_r + self.gamma * (1.0 - d) * V_next         # [B]

        value_loss_main = self.value_criterion(V_pred, target_return)

        # ------- Filtered critic + reconstruction + KL + filter reg -------
        # Decode others' info
        recon_target = torch.cat([others_obs, others_act], dim=-1)        # [B, others_obs_dim + others_act_dim]
        # decoder is already on correct device from __init__, no need to move
        recon = decoder(belief)                                           # [B, ...] on same device as belief

        # filt is already on correct device from __init__, no need to move
        phi = filt()  # [features] on device
        # Explicitly ensure phi is on the same device as other tensors
        phi = phi.to(device)
        # broadcast to batch
        weighted_error = (recon_target - recon) ** 2 * phi.unsqueeze(0)
        recon_loss = weighted_error.mean()

        # KL(q(b|x)||p(b)) with prior N(0, I)
        kl_element = torch.exp(logvar) + mu**2 - 1.0 - logvar
        kl_loss = 0.5 * kl_element.sum(dim=-1).mean()

        filter_reg = 0.5 * (phi**2).sum()

        # filtered critic uses filtered others' obs
        others_obs_dim = self.others_obs_dims[agent_id]
        phi_obs = phi[:others_obs_dim]
        filt_others_obs = others_obs * phi_obs.unsqueeze(0)               # [B, others_obs_dim]
        filt_state = torch.cat([own_obs, filt_others_obs], dim=-1)        # [B, obs_dim + others_obs_dim]
        # critic_filt is already on correct device from __init__
        V_filt = critic_filt(filt_state).squeeze(-1)
        value_loss_filt = self.value_criterion(V_filt, target_return)

        # ------- Actor loss (advantage actor-critic) -------
        with torch.no_grad():
            V_detached = V_pred.detach()
            advantage = (target_return - V_detached)

        # actor is already on correct device from beginning of _learn_agent
        actor_input = torch.cat([own_obs, belief.detach()], dim=-1).to(device)
        logits = actor(actor_input)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(own_actions)
        actor_loss = -(log_probs * advantage).mean()

        # ------- Total loss -------
        total_loss = (
            actor_loss
            + value_loss_main
            + value_loss_filt
            + self.recon_coef * recon_loss
            + self.kl_coef * kl_loss
            + self.filter_reg_coef * filter_reg
        )

        # ------- Backprop -------
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        filter_opt.zero_grad()
        actor_opt.zero_grad()
        critic_main_opt.zero_grad()
        critic_filt_opt.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()

        encoder_opt.step()
        decoder_opt.step()
        filter_opt.step()
        actor_opt.step()
        critic_main_opt.step()
        critic_filt_opt.step()

        return float(total_loss.item())


    # ==========================================================
    #   Evaluation (copied / simplified from MATD3)
    # ==========================================================

    def test(
        self,
        env: PzEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        sum_scores: bool = True,
    ) -> float:
        """Evaluate SMPE policy (greedy) on env."""

        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            if hasattr(env, "num_envs"):
                num_envs = env.num_envs
                is_vectorised = True
            else:
                num_envs = 1
                is_vectorised = False

            for _ in range(loop):
                obs, info = env.reset()
                scores = (
                    np.zeros((num_envs, 1))
                    if sum_scores
                    else np.zeros((num_envs, len(self.agent_ids)))
                )
                completed_episode_scores = (
                    np.zeros((num_envs, 1))
                    if sum_scores
                    else np.zeros((num_envs, len(self.agent_ids)))
                )
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    step += 1
                    if swap_channels:
                        expand_dims = not is_vectorised
                        obs = {
                            agent_id: obs_channels_to_first(s, expand_dims)
                            for agent_id, s in obs.items()
                        }

                    action, _ = self.get_action(
                        obs,
                        infos=info,
                    )

                    if not is_vectorised:
                        action = {agent: act[0] for agent, act in action.items()}

                    obs, reward, term, trunc, info = env.step(action)

                    agent_rewards = np.array(list(reward.values())).transpose()
                    agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                    score_increment = (
                        (
                            np.sum(agent_rewards, axis=-1)[:, np.newaxis]
                            if is_vectorised
                            else np.sum(agent_rewards, axis=-1)
                        )
                        if sum_scores
                        else agent_rewards
                    )
                    scores += score_increment

                    dones = {}
                    for agent_id in self.agent_ids:
                        terminated = term.get(agent_id, True)
                        truncated = trunc.get(agent_id, False)

                        terminated = np.where(
                            np.isnan(terminated), True, terminated
                        ).astype(bool)
                        truncated = np.where(
                            np.isnan(truncated), False, truncated
                        ).astype(bool)

                        dones[agent_id] = terminated | truncated

                    if not is_vectorised:
                        dones = {
                            agent: np.array([dones[agent_id]])
                            for agent in self.agent_ids
                        }

                    for idx, agent_dones in enumerate(zip(*dones.values())):
                        if (
                            np.all(agent_dones)
                            or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1

                rewards.append(np.mean(completed_episode_scores, axis=0))

        mean_fit = np.mean(rewards, axis=0)
        mean_fit = mean_fit[0] if sum_scores else mean_fit
        self.fitness.append(mean_fit)
        return float(mean_fit)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # Initialize filter optimizers after registry init
        agent._init_filter_optimizers()
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
    evo_steps = 10_000     # Evolution frequency
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
    start_time = datetime.now()
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

    # Create experiment directory matching other experiments
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = f"./results/{exp_id}"
    os.makedirs(path, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(path, f"{exp_id}_model.pt")
    elite.save_checkpoint(model_path)
    print(f"Model saved: {model_path}")

    # Save training scores data
    scores_data_path = os.path.join(path, f"{exp_id}_data.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    print(f"Training data saved: {scores_data_path}")

    # Plot and save training scores
    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Training Progress - SMPE', fontsize=14)
    plt.xlabel('Evolution Iterations', fontsize=12)
    plt.ylabel('Mean Population Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(path, f"{exp_id}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()

    # Save metrics.json
    metrics = {
        "final_score": float(training_scores_history[-1]),
        "best_score": float(max(training_scores_history)),
        "worst_score": float(min(training_scores_history)),
        "total_iterations": len(training_scores_history),
        "hyperparameters": {k: v for k, v in INIT_HP.items() if k != "AGENT_IDS"},
        "network_config": NET_CONFIG,
        "training": {
            "max_steps": max_steps,
            "num_envs": num_envs,
            "evo_steps": evo_steps,
            "device": device,
        }
    }
    metrics_path = os.path.join(path, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Copy config file to results directory
    config_path = "configs/experiments/smpe_baseline.yaml"
    if Path(config_path).exists():
        shutil.copy(config_path, os.path.join(path, "config.yaml"))

    # Register in experiments.csv
    end_time = datetime.now()
    duration_hours = (end_time - start_time).total_seconds() / 3600
    registry_path = "./results/experiments.csv"

    # Create CSV if it doesn't exist
    if not Path(registry_path).exists():
        with open(registry_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "name", "start_time", "end_time", "status",
                "steps", "duration_hours", "final_score", "best_score",
                "worst_score", "config_path"
            ])

    # Append experiment entry
    with open(registry_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_id,
            "smpe_baseline",
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed",
            total_steps,
            f"{duration_hours:.2f}",
            f"{metrics['final_score']:.2f}",
            f"{metrics['best_score']:.2f}",
            f"{metrics['worst_score']:.2f}",
            config_path,
        ])

    print(f"\nResults saved to: {path}")
    print(f"Registered in: {registry_path}")

    pbar.close()
    env.close()
