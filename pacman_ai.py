"""Spectate two Ms. Pac-Man DQN checkpoints side by side.

  .venv/Scripts/python.exe pacman_ai.py
  .venv/Scripts/python.exe pacman_ai.py --a models/mspacman/dqn_envpool/final_4000000.zip \
      --b models/mspacman/dqn_envpool/final_6000000.zip

Both panes use the exact preprocessing each model was trained and evaluated with, so the
scores on screen are comparable to the recorded numbers (4M = 2,355, 10M = 1,317).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch as th
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacman_duel import EnvPoolObs, RainbowBot  # noqa: E402

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent
ENV_ID = "ALE/MsPacman-v5"


class Pane:
    """One side of the AI-vs-AI race. A .pt path loads the C51 Rainbow net (EnvPool-order
    observations); a .zip path loads the older SB3 DQN through AtariPreprocessing."""

    def __init__(self, model_path: Path, label: str, seed: int, device: str):
        self.rainbow = model_path.suffix == ".pt"
        if self.rainbow:
            self.bot = RainbowBot(model_path, device)
            e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
            self.env = EnvPoolObs(e)
        else:
            self.model = DQN.load(str(model_path), device="cpu")
            self.model.replay_buffer = None
            self.model.policy.to(device)
            e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
            e = AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True,
                                   grayscale_newaxis=False, scale_obs=False,
                                   terminal_on_life_loss=False)
            self.env = FrameStack(e, 4)
        self.label = label
        self.seed = seed
        self.best = 0.0
        self.rounds = 0
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset(seed=self.seed + self.rounds)
        self.score = 0.0

    def step(self) -> np.ndarray:
        if self.rainbow:
            self.obs, r, term, trunc, _ = self.env.step(self.bot.act(self.obs))
        else:
            a, _ = self.model.predict(np.asarray(self.obs)[None].astype(np.uint8), deterministic=True)
            self.obs, r, term, trunc, _ = self.env.step(int(np.asarray(a).reshape(-1)[0]))
        self.score += float(r)
        if term or trunc:
            self.best = max(self.best, self.score)
            self.rounds += 1
            self.reset()
        return self.env.render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="models/rainbow_scale_10m/best.pt")
    ap.add_argument("--b", default="models/mspacman/dqn_envpool/final_4000000.zip")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    lab_a = "Rainbow / 3,230" if args.a.endswith(".pt") else "DQN / 2,355"
    lab_b = "Rainbow / 3,230" if args.b.endswith(".pt") else "DQN / 2,355"
    pa = Pane(ROOT / args.a, lab_a, 7, device)
    pb = Pane(ROOT / args.b, lab_b, 107, device)

    pygame.init()
    font = pygame.font.SysFont("consolas", 13)
    big = pygame.font.SysFont("consolas", 18, bold=True)
    screen = None
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
        fa, fb = pa.step(), pb.step()
        sa = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(fa, (1, 0, 2))),
                                    (160 * args.scale, 210 * args.scale))
        sb = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(fb, (1, 0, 2))),
                                    (160 * args.scale, 210 * args.scale))
        if screen is None:
            screen = pygame.display.set_mode((sa.get_width() * 2 + 12, sa.get_height() + 74))
            pygame.display.set_caption("Ms. Pac-Man — bot vs bot")
        screen.fill((10, 10, 16))
        screen.blit(sa, (0, 0)); screen.blit(sb, (sa.get_width() + 12, 0))
        y = sa.get_height() + 4
        screen.blit(big.render(f"A {pa.label:<16} {int(pa.score):>5}  best {int(pa.best):>5}",
                               True, (34, 211, 238)), (6, y))
        screen.blit(big.render(f"B {pb.label:<16} {int(pb.score):>5}  best {int(pb.best):>5}",
                               True, (250, 204, 21)),
                    (sa.get_width() + 18, y))
        screen.blit(font.render(f"best A {int(pa.best)} · best B {int(pb.best)} · "
                                f"episodes {pa.rounds}/{pb.rounds} · ESC quit", True, (130, 140, 155)), (6, y + 28))
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
