"""Gymnasium wrapper around ViZDoom scenarios.

Observation: 84x84 uint8 grayscale (channel-last) -> stack with VecFrameStack.
Action: discrete over every combination of the scenario's available buttons
        (defend_the_center: 3 buttons -> 8 actions).
Reward: the scenario's own reward (kills +1, death -1, ...).

  python doom_env.py            # smoke test: random policy, prints fps
"""
from __future__ import annotations

import itertools
import os
from typing import Optional, Sequence

# keep the discrete GPU / software renderer honest
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

SCENARIOS = {
    "defend_the_center": "defend_the_center.cfg",
    "defend_the_line": "defend_the_line.cfg",
    "basic": "basic.cfg",
    "deadly_corridor": "deadly_corridor.cfg",
    "health_gathering": "health_gathering.cfg",
    "health_gathering_supreme": "health_gathering_supreme.cfg",
    "my_way_home": "my_way_home.cfg",
    "take_cover": "take_cover.cfg",
    "predict_position": "predict_position.cfg",
    "deathmatch": "deathmatch.cfg",
}


class DoomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 35}

    def __init__(
        self,
        scenario: str = "defend_the_center",
        frame_skip: int = 4,
        screen_size: int = 84,
        resolution: tuple[int, int] = (160, 120),
        render_mode: Optional[str] = None,
        buttons: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = 1000,
        screen_format: str = "GRAY8",
        single_button_actions: bool = True,
    ) -> None:
        super().__init__()
        self.single_button_actions = single_button_actions
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; options: {sorted(SCENARIOS)}")
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.resolution = resolution
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._seed = seed

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, SCENARIOS[scenario]))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24 if screen_format == "RGB24" else vzd.ScreenFormat.GRAY8)
        self.game.set_window_visible(render_mode == "human")
        self.game.set_console_enabled(False)
        self.game.set_sound_enabled(False)
        if seed is not None:
            self.game.set_seed(seed)
        if buttons is not None:
            self.game.set_available_buttons(buttons)
        self.game.init()

        n_btns = len(self.game.get_available_buttons())
        self.button_names = [str(b).split(".")[-1] for b in self.game.get_available_buttons()]
        if self.single_button_actions:
            # one button at a time plus NOOP: a 3x smaller action space, which is the
            # difference between "learns to aim" and "wanders" on defend_the_center
            combos = [[0] * n_btns] + [[1 if j == i else 0 for j in range(n_btns)] for i in range(n_btns)]
        else:
            combos = list(itertools.product([0, 1], repeat=n_btns))
        self.actions = np.array(combos, dtype=np.uint8)

        self.observation_space = spaces.Box(0, 255, (screen_size, screen_size, 1), np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))
        self._steps = 0
        self._last_rgb = np.zeros((resolution[1], resolution[0], 3), np.uint8)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        state = self.game.get_state()
        buf = state.screen_buffer if state is not None else None
        if buf is None:  # episode over, no frame available
            self._last_rgb = np.zeros((self.resolution[1], self.resolution[0], 3), np.uint8)
            return np.zeros((self.screen_size, self.screen_size, 1), np.uint8)
        if buf.ndim == 3:  # RGB24: (H, W, 3)
            self._last_rgb = buf.copy()
            gray = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
        else:
            self._last_rgb = np.repeat(buf[..., None], 3, axis=2)
            gray = buf
        frame = cv2.resize(gray, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return frame[..., None].astype(np.uint8)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None and seed != self._seed:
            self.game.set_seed(seed)
            self._seed = seed
        self.game.new_episode()
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        reward = self.game.make_action(self.actions[int(action)].tolist(), self.frame_skip)
        self._steps += 1
        finished = self.game.is_episode_finished()
        dead = finished and self.game.is_player_dead()
        obs = self._obs()
        truncated = bool(finished and not dead)
        if self.max_steps is not None and self._steps >= self.max_steps and not finished:
            truncated = True
        term = bool(dead and not truncated)
        return obs, float(reward), term, truncated, {}

    def render(self):
        return self._last_rgb if self.render_mode == "rgb_array" else None

    def rgb_frame(self) -> np.ndarray:
        return self._last_rgb

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass

    # convenience for eval scripts
    def action_meanings(self):
        return ["+".join(n for n, v in zip(self.button_names, a) if v) or "NOOP" for a in self.actions]


if __name__ == "__main__":
    import time

    sc = "defend_the_center"
    env = DoomEnv(sc)
    print("scenario:", sc, "buttons:", env.button_names, "actions:", env.action_space.n)
    print("obs space:", env.observation_space)
    t0, n, eps = time.time(), 0, 0
    obs, _ = env.reset()
    for _ in range(3000):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            eps += 1
            obs, _ = env.reset()
    dt = time.time() - t0
    print(f"{n} random steps in {dt:.1f}s = {n / dt:.0f} steps/s, episodes={eps}, obs={obs.shape} {obs.dtype}")
    env.close()
