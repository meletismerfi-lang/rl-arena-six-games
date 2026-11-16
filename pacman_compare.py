"""Head-to-head pygame viewer: two Ms. Pac-Man checkpoints playing in parallel.

  .venv/Scripts/python.exe pacman_compare.py
  .venv/Scripts/python.exe pacman_compare.py --a models/mspacman/dqn_envpool/final_4000000.zip \
      --b models/runpod/final_30000000.zip --scale 2

Both players get the same ALE seed, so the maze and ghost RNG are identical — any score
difference is the policy, not luck. Keys: SPACE pause · R restart · S shot · ESC quit.

Verification without a display: --headless N --shot out.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import AtariPreprocessing, FrameStack

import ale_py

gym.register_envs(ale_py)

ROOT = Path(__file__).resolve().parent
ENV_ID = "ALE/MsPacman-v5"

DEFAULT_A = "models/mspacman/dqn_envpool/final_7000000.zip"      # 20M steps, mean 1350
DEFAULT_B = "models/runpod/final_30000000.zip"                    # 30M steps (pod), mean 1042
LABEL_A = "20M steps · 32 envs"
LABEL_B = "30M steps · 256 envs (pod)"


def load_policy(path: Path, device: str):
    for cls in (DQN, PPO):
        try:
            model = cls.load(str(path), device="cpu")
            break
        except Exception:
            continue
    else:
        raise SystemExit(f"cannot load {path}")
    model.replay_buffer = None          # never needed for inference; it is GBs of VRAM
    model.policy.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


class Player:
    def __init__(self, path: Path, label: str, device: str, seed: int):
        self.model = load_policy(path, device)
        self.label = label
        self.path = path
        self.env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
        self.env = AtariPreprocessing(self.env, frame_skip=4, screen_size=84, grayscale_obs=True,
                                      grayscale_newaxis=False, scale_obs=False,
                                      terminal_on_life_loss=False)
        self.env = FrameStack(self.env, 4)
        self.seed = seed
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset(seed=self.seed)
        self.score = 0.0
        self.finished = 0

    def step(self) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(self.obs)[None].astype(np.uint8), deterministic=True)
        self.obs, reward, term, trunc, _ = self.env.step(int(np.asarray(action).reshape(-1)[0]))
        self.score += float(reward)
        if term or trunc:
            self.finished += 1
            self.reset()
        return self.env.render()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", default=DEFAULT_A)
    p.add_argument("--b", default=DEFAULT_B)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--headless", type=int, default=0)
    p.add_argument("--shot", default=None)
    args = p.parse_args()

    a_path = Path(args.a) if Path(args.a).is_absolute() else ROOT / args.a
    b_path = Path(args.b) if Path(args.b).is_absolute() else ROOT / args.b
    for f in (a_path, b_path):
        if not f.exists():
            raise SystemExit(f"missing model: {f}")

    print(f"A: {a_path.name} ({LABEL_A})")
    print(f"B: {b_path.name} ({LABEL_B})")
    A = Player(a_path, LABEL_A, args.device, args.seed)
    B = Player(b_path, LABEL_B, args.device, args.seed)   # same seed on purpose

    pygame.init()
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 19, bold=True)
    screen = None
    clock = pygame.time.Clock()
    frame_a = A.env.render()
    frame_b = B.env.render()
    paused = False
    running = True
    steps = 0

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    A.reset(); B.reset()
                elif ev.key == pygame.K_s and screen is not None:
                    out = ROOT / "videos" / "compare_shot.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(screen, str(out))
                    print("screenshot:", out)

        if not paused:
            frame_a = A.step()
            frame_b = B.step()
            steps += 1

        h, w = frame_a.shape[:2]
        sa = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(frame_a, (1, 0, 2))),
                                    (w * args.scale, h * args.scale))
        sb = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(frame_b, (1, 0, 2))),
                                    (w * args.scale, h * args.scale))
        gap = 12
        hud = 74
        win = (sa.get_width() + sb.get_width() + gap, sa.get_height() + hud)
        if screen is None or screen.get_size() != win:
            screen = pygame.display.set_mode(win)
            pygame.display.set_caption("Ms. Pac-Man — 20M vs 30M (DQN)")
        screen.fill((10, 10, 16))
        screen.blit(sa, (0, 0))
        screen.blit(sb, (sa.get_width() + gap, 0))

        lead = "A" if A.score > B.score else ("B" if B.score > A.score else "=")
        screen.blit(big.render(f"A  {A.label:<26} score {A.score:6.0f}", True, (34, 211, 238)), (8, sa.get_height() + 4))
        screen.blit(big.render(f"B  {B.label:<26} score {B.score:6.0f}", True, (250, 204, 21)), (8, sa.get_height() + 28))
        screen.blit(font.render(f"leader: {lead}   episodes A/B: {A.finished}/{B.finished}   "
                                f"SPACE pause · R reset · S shot · ESC quit", True, (120, 130, 145)),
                    (8, sa.get_height() + 54))
        if paused:
            t = big.render("PAUSED", True, (250, 204, 21))
            screen.blit(t, (win[0] // 2 - t.get_width() // 2, 8))
        pygame.display.flip()

        if args.headless:
            if steps >= args.headless:
                if args.shot:
                    out = Path(args.shot)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(screen, str(out))
                    print(f"screenshot: {out}  A={A.score:.0f} B={B.score:.0f}")
                running = False
            time.sleep(1 / args.fps)
        else:
            clock.tick(args.fps)

    A.env.close(); B.env.close()
    if not args.headless:
        pygame.quit()


if __name__ == "__main__":
    main()
