"""Replay buffer that lives in GPU VRAM instead of system RAM.

SB3's ReplayBuffer keeps observations in numpy on the host (100k x 4 x 84 x 84
uint8 + next_obs copy ~ 5.6 GB RAM) and copies every minibatch over PCIe.
This one keeps the frames in a single uint8 tensor on the GPU
(100k x 4 x 84 x 84 ~ 2.8 GB VRAM), reuses the next frame in-place
(optimize-memory style) and samples directly on-device.

Drop-in for DQN:
    DQN(..., replay_buffer_class=TorchReplayBuffer, replay_buffer_kwargs={"pin_obs": False})
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class TorchReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "cuda",
        n_envs: int = 1,
        optimize_memory_usage: bool = True,   # always single-array here
        handle_timeout_termination: bool = True,
        pin_obs: bool = False,                # keep True to store frames in pinned host RAM instead
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.optimize_memory_usage = True
        self.handle_timeout_termination = handle_timeout_termination
        self.pin_obs = pin_obs
        dev = th.device(self.device)
        storage_dev = th.device("cpu") if pin_obs else dev
        self.obs = th.zeros((buffer_size, n_envs, *self.obs_shape), dtype=th.uint8, device=storage_dev)
        if pin_obs:
            self.obs = self.obs.pin_memory()
        self.actions = th.zeros((buffer_size, n_envs, self.action_dim), dtype=th.int64, device=dev)
        self.rewards = th.zeros((buffer_size, n_envs), dtype=th.float32, device=dev)
        self.dones = th.zeros((buffer_size, n_envs), dtype=th.float32, device=dev)
        self.timeouts = th.zeros((buffer_size, n_envs), dtype=th.float32, device=dev)

    # ---- writes -----------------------------------------------------------
    def add(self, obs, next_obs, action, reward, done, infos: List[Dict[str, Any]]) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))
        t = th.as_tensor
        self.obs[self.pos] = t(np.asarray(obs), dtype=th.uint8, device=self.obs.device)
        self.actions[self.pos] = t(np.asarray(action), device=self.actions.device)
        self.rewards[self.pos] = t(np.asarray(reward), dtype=th.float32, device=self.rewards.device)
        self.dones[self.pos] = t(np.asarray(done), dtype=th.float32, device=self.dones.device)
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = t(
                np.array([bool(i.get("TimeLimit.truncated", False)) for i in infos], dtype=np.float32),
                device=self.timeouts.device,
            )
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if isinstance(batch_inds, th.Tensor):
            idx = batch_inds.to(dtype=th.long, device=self.obs.device)
        else:
            idx = th.as_tensor(np.asarray(batch_inds), dtype=th.long, device=self.obs.device)
        nxt = (idx + 1) % self.buffer_size
        obs = self.obs[idx].reshape(-1, *self.obs_shape)
        next_obs = self.obs[nxt].reshape(-1, *self.obs_shape)
        if obs.device.type != "cuda" or self.device != "cuda":
            obs, next_obs = obs.to(self.device), next_obs.to(self.device)
        dones = self.dones[idx].reshape(-1, 1)
        if self.handle_timeout_termination:
            dones = dones * (1.0 - self.timeouts[idx].reshape(-1, 1))
        actions = self.actions[idx].reshape(-1, self.action_dim)
        return ReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=self.rewards[idx].reshape(-1, 1),
        )

    # ---- reads ------------------------------------------------------------
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.full:
            idx = (th.randint(1, self.buffer_size, (batch_size,), device=self.obs.device) + self.pos) % self.buffer_size
        else:
            idx = th.randint(0, self.pos, (batch_size,), device=self.obs.device)
        return self._get_samples(idx, env=env)

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def __len__(self) -> int:
        return self.size()

    def vram_gb(self) -> float:
        if self.obs.device.type != "cuda":
            return 0.0
        n = self.obs.numel() + self.actions.numel() * 8 + self.rewards.numel() * 4 * 3
        return n / 1024**3
