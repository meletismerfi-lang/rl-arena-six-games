"""Ms. Pac-Man head to head: you with the arrow keys, the bot beside you.

  .venv/Scripts/python.exe pacman_duel.py                 # Rainbow C51 (3,230 on this harness)
  .venv/Scripts/python.exe pacman_duel.py --agent dqn      # the older DQN (2,355)

Both play their own copy of the same game with the same rules (4-frame skip, no sticky
actions). The Rainbow bot feeds its network exactly the observation EnvPool produced during
training: grayscale first, then INTER_AREA resize to 84x84, max-pooled over the last two of the
four skipped frames. That ordering was verified against EnvPool output pixel by pixel (99.6%
of pixels identical) and the bot scores 3,230 here against 3,530 on EnvPool.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import pygame
import torch as th
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rainbow_pacman_3 import DuelingC51  # noqa: E402

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent
ENV_ID = "ALE/MsPacman-v5"
MODEL_DQN = ROOT / "models" / "mspacman" / "dqn_envpool" / "final_4000000.zip"
MODEL_RAINBOW = ROOT / "models" / "rainbow_scale_10m" / "best.pt"
VMIN, VMAX, ATOMS = -1.0, 25.0, 51
N_ACT = 9

# ALE's Ms. Pac-Man action order
NOOP, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
KEYMAP = {pygame.K_UP: UP, pygame.K_w: UP, pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
          pygame.K_LEFT: LEFT, pygame.K_a: LEFT, pygame.K_DOWN: DOWN, pygame.K_s: DOWN}


def human_env(seed: int):
    e = gym.make(ENV_ID, frameskip=4, repeat_action_probability=0.0, render_mode="rgb_array")
    e.reset(seed=seed)
    return e


def prep_obs(rgb: np.ndarray) -> np.ndarray:
    """EnvPool's order: grayscale, then INTER_AREA resize. Verified 99.6% pixel-exact."""
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)


class EnvPoolObs(gym.Wrapper):
    """Four-frame skip with the last two frames max-pooled, stacked four deep.

    This reproduces what envpool's MsPacman-v5 emitted during training, which is the only
    observation the Rainbow network has ever seen.
    """

    def __init__(self, env, skip: int = 4, depth: int = 4):
        super().__init__(env)
        self.skip, self.depth = skip, depth
        self.stack: deque = deque([np.zeros((84, 84), np.uint8)] * depth, maxlen=depth)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        small = prep_obs(frame)
        self.stack = deque([small] * self.depth, maxlen=self.depth)
        return np.stack(self.stack), info

    def step(self, action):
        frames, reward, done, info = [], 0.0, False, {}
        for _ in range(self.skip):
            frame, r, term, trunc, info = self.env.step(int(action))
            frames.append(frame)
            reward += float(r)
            if term or trunc:
                done = True
                break
        pooled = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
        self.stack.append(prep_obs(pooled))
        return np.stack(self.stack), reward, done, False, info


class RainbowBot:
    """Greedy C51 policy over the distributional expectation."""

    def __init__(self, path: Path, device: str):
        self.device = device
        self.net = DuelingC51(4, N_ACT, ATOMS, width=64).to(device).eval()
        self.net.load_state_dict(th.load(str(path), map_location=device, weights_only=True))
        self.support = th.linspace(VMIN, VMAX, ATOMS, device=device)

    def act(self, obs_stack: np.ndarray) -> int:
        with th.no_grad():
            q = self.net.q(th.as_tensor(np.asarray(obs_stack)[None], device=self.device), self.support)
        return int(q.argmax(1).item())


class DQNBot:
    """The earlier SB3 DQN, kept so the two agents can be compared in the same window."""

    def __init__(self, path: Path):
        self.model = DQN.load(str(path), device="cpu")
        self.model.replay_buffer = None
        self.model.policy.to("cuda" if th.cuda.is_available() else "cpu")

    def act(self, obs_stack: np.ndarray) -> int:
        action, _ = self.model.predict(np.asarray(obs_stack)[None].astype(np.uint8), deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])


def bot_env(seed: int, agent: str):
    if agent == "rainbow":
        e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
        e = EnvPoolObs(e)
    else:
        e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
        e = AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True,
                              grayscale_newaxis=False, scale_obs=False, terminal_on_life_loss=False)
        e = FrameStack(e, 4)
    e.reset(seed=seed)
    return e


def load_bot(agent: str):
    if agent == "rainbow":
        return RainbowBot(MODEL_RAINBOW, "cuda" if th.cuda.is_available() else "cpu")
    return DQNBot(MODEL_DQN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent", default="rainbow", choices=["rainbow", "dqn"])
    args = ap.parse_args()

    bot = load_bot(args.agent)
    h_env, b_env = human_env(args.seed), bot_env(args.seed + 100, args.agent)
    h_obs, _ = h_env.reset(seed=args.seed)
    b_obs, _ = b_env.reset(seed=args.seed + 100)

    pygame.init()
    font = pygame.font.SysFont("consolas", 14)
    big = pygame.font.SysFont("consolas", 20, bold=True)
    screen = None
    clock = pygame.time.Clock()

    h_score = b_score = 0.0
    h_lives = b_lives = 3
    rounds = {"you": 0, "bot": 0}
    action = NOOP
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in KEYMAP:
                    action = KEYMAP[ev.key]
                elif ev.key == pygame.K_r:
                    h_obs, _ = h_env.reset(seed=args.seed)
                    b_obs, _ = b_env.reset(seed=args.seed + 100)
                    h_score = b_score = 0
                    h_lives = b_lives = 3
                    action = NOOP

        h_obs, h_r, h_term, h_trunc, h_info = h_env.step(action)
        h_score += float(h_r)
        h_lives = int(h_info.get("lives", h_lives))

        b_action = bot.act(b_obs)
        b_obs, b_r, b_term, b_trunc, b_info = b_env.step(b_action)
        b_score += float(b_r)
        b_lives = int(b_info.get("lives", b_lives))

        if h_term or h_trunc:
            rounds["you" if h_score > b_score else "bot"] += 1
            h_obs, _ = h_env.reset(seed=args.seed + rounds["you"] + rounds["bot"])
            h_score, h_lives = 0, 3
        if b_term or b_trunc:
            rounds["you" if h_score > b_score else "bot"] += 1
            b_obs, _ = b_env.reset(seed=args.seed + 100 + rounds["you"] + rounds["bot"])
            b_score, b_lives = 0, 3

        fa = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(h_env.render(), (1, 0, 2))),
            (160 * args.scale, 210 * args.scale))
        fb = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(b_env.render(), (1, 0, 2))),
            (160 * args.scale, 210 * args.scale))
        if screen is None:
            screen = pygame.display.set_mode((fa.get_width() * 2 + 12, fa.get_height() + 74))
            pygame.display.set_caption("Ms. Pac-Man — you vs the DQN")
        screen.fill((10, 10, 16))
        screen.blit(fa, (0, 0))
        screen.blit(fb, (fa.get_width() + 12, 0))
        y = fa.get_height() + 4
        screen.blit(big.render(f"YOU {int(h_score):>5}", True, (34, 211, 238)), (6, y))
        screen.blit(big.render(f"BOT {int(b_score):>5}", True, (250, 204, 21)), (fa.get_width() + 18, y))
        screen.blit(font.render(f"lives {h_lives} · rounds won you {rounds['you']} / bot {rounds['bot']} "
                                f"· arrows to move · R restart · ESC quit", True, (130, 140, 155)), (6, y + 28))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
