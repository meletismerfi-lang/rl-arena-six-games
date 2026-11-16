"""Ms. Pac-Man head to head: you with the arrow keys, the DQN beside you.

  .venv/Scripts/python.exe pacman_duel.py

Both play their own copy of the same game with the same rules (4-frame skip, no sticky
actions). The bot uses exactly the preprocessing it was trained and evaluated with, so its
score here matches the 2,355 mean on record.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch as th
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import DQN

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent
ENV_ID = "ALE/MsPacman-v5"
MODEL = ROOT / "models" / "mspacman" / "dqn_envpool" / "final_4000000.zip"

# ALE's Ms. Pac-Man action order
NOOP, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
KEYMAP = {pygame.K_UP: UP, pygame.K_w: UP, pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
          pygame.K_LEFT: LEFT, pygame.K_a: LEFT, pygame.K_DOWN: DOWN, pygame.K_s: DOWN}


def human_env(seed: int):
    e = gym.make(ENV_ID, frameskip=4, repeat_action_probability=0.0, render_mode="rgb_array")
    e.reset(seed=seed)
    return e


def bot_env(seed: int):
    e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
    e = AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True,
                           grayscale_newaxis=False, scale_obs=False, terminal_on_life_loss=False)
    e = FrameStack(e, 4)
    e.reset(seed=seed)
    return e


def load_bot():
    model = DQN.load(str(MODEL), device="cpu")
    model.replay_buffer = None
    model.policy.to("cuda" if th.cuda.is_available() else "cpu")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model = load_bot()
    h_env, b_env = human_env(args.seed), bot_env(args.seed + 100)
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

        b_action, _ = model.predict(np.asarray(b_obs)[None].astype(np.uint8), deterministic=True)
        b_obs, b_r, b_term, b_trunc, b_info = b_env.step(int(np.asarray(b_action).reshape(-1)[0]))
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
