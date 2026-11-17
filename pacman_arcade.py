"""Pygame arcade for the trained Ms. Pac-Man agent.

  .venv/Scripts/python.exe pacman_arcade.py --model models/mspacman/dqn_envpool/final_4000000.zip
  .venv/Scripts/python.exe pacman_arcade.py --model models/mspacman/ppo_envpool/final.zip --scale 3

Keys
  TAB    toggle agent / human control
  UP DOWN LEFT RIGHT   steer when in human mode
  SPACE  pause        R  new episode       S  screenshot
  ESC / Q  quit

Rendering: the ALE frame is blitted at `--scale`; the policy sees exactly what it was
trained on — 84x84 grayscale, 4 stacked frames, 4-frame action repeat.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
from stable_baselines3 import DQN, PPO

import ale_py

gym.register_envs(ale_py)

ROOT = Path(__file__).resolve().parent
ALE_MSPACMAN = "ALE/MsPacman-v5"

# ALE Ms. Pac-Man action order: NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
HUMAN_KEYS = {
    pygame.K_UP: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_DOWN: 4,
}


def load_policy(path: Path, device: str):
    """Load for inference without the training replay buffer taking over VRAM.

    DQN.load(device='cuda') also materialises the saved replay-buffer spec — a 200k-frame
    GPU allocation (5.4 GB) that inference never uses. Build on CPU, drop the buffer,
    then move just the network.
    """
    for cls in (DQN, PPO):
        try:
            model = cls.load(str(path), device="cpu")
            break
        except Exception:
            continue
    else:
        raise SystemExit(f"could not load {path} as DQN or PPO")

    model.replay_buffer = None
    try:
        model.observation_space, model.action_space  # noqa: B018
        model.policy.to(device)
        model.device = torch.device(device)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


class PacmanArcade:
    def __init__(self, model_path: Path, scale: int = 3, seed: int = 7, device: str = "cuda",
                 fps: int = 30, headless: int = 0, shot: str | None = None):
        self.model = load_policy(model_path, device)
        self.model_name = f"{type(self.model).__name__} · {model_path.parent.name}/{model_path.stem}"
        self.scale = scale
        self.fps = fps
        self.headless = headless
        self.shot = shot

        self.env = gym.make(ALE_MSPACMAN, frameskip=1, repeat_action_probability=0.0,
                            render_mode="rgb_array")
        from gymnasium.wrappers import AtariPreprocessing, FrameStack
        self.env = AtariPreprocessing(self.env, frame_skip=4, screen_size=84, grayscale_obs=True,
                                      grayscale_newaxis=False, scale_obs=False,
                                      terminal_on_life_loss=False)
        self.env = FrameStack(self.env, 4)
        self._obs, _ = self.env.reset(seed=seed)
        self.agent_mode = True
        self.paused = False
        self.episode_return = 0.0
        self.episodes = 0
        self.steps = 0
        self.best = 0.0
        self.fps_hist: deque[float] = deque(maxlen=30)

        self.screen = None
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 20, bold=True)
        if not headless:
            pygame.display.set_caption(f"Ms. Pac-Man RL — {self.model_name}")

    # ---------------------------------------------------------------- helpers
    def policy_obs(self) -> np.ndarray:
        """FrameStack already gives (4, 84, 84); SB3 wants a leading batch dim."""
        return np.asarray(self._obs)[None].astype(np.uint8)

    def reset_episode(self, seed: int | None = None):
        self._obs, _ = self.env.reset(seed=seed)
        self.episode_return = 0.0
        self.steps = 0

    def draw(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (w * self.scale, h * self.scale))

        mode = "AGENT" if self.agent_mode else "HUMAN"
        fps = np.mean(self.fps_hist) if self.fps_hist else 0.0
        lines = [
            (f"{mode}   score {self.episode_return:6.0f}   best {self.best:5.0f}   ep {self.episodes}",
             self.big, (34, 211, 238)),
            (self.model_name, self.font, (148, 163, 184)),
            (f"{fps:4.1f} fps   TAB agent/human · R reset · SPACE pause · S shot · ESC quit",
             self.font, (100, 116, 139)),
        ]
        widths = [font.size(text)[0] for text, font, _ in lines]
        win_w = max(surf.get_width(), max(widths) + 16)
        hud_h = sum(font.get_height() + 3 for _, font, _ in lines) + 8

        if self.screen is None or self.screen.get_size() != (win_w, surf.get_height() + hud_h):
            self.screen = pygame.display.set_mode((win_w, surf.get_height() + hud_h))
        self.screen.fill((10, 10, 16))
        self.screen.blit(surf, (0, 0))

        y = surf.get_height() + 4
        for text, font, color in lines:
            self.screen.blit(font.render(text, True, color), (8, y))
            y += font.get_height() + 3
        if self.paused:
            p = self.big.render("PAUSED", True, (250, 204, 21))
            self.screen.blit(p, (self.screen.get_width() // 2 - p.get_width() // 2, 12))
        pygame.display.flip()

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        frame = self.env.render()
        running = True
        target_dt = 1.0 / self.fps
        last = time.time()
        total_steps = 0          # never reset by episode boundaries (self.steps is)
        while running:
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                self.fps_hist.append(1.0 / dt)

            if not self.headless:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif ev.key == pygame.K_TAB:
                            self.agent_mode = not self.agent_mode
                        elif ev.key == pygame.K_r:
                            self.reset_episode()
                        elif ev.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif ev.key == pygame.K_s and self.screen is not None:
                            out = ROOT / "videos" / "arcade_shot.png"
                            out.parent.mkdir(parents=True, exist_ok=True)
                            pygame.image.save(self.screen, str(out))
                            print("screenshot:", out)

            if not self.paused:
                if self.agent_mode:
                    action, _ = self.model.predict(self.policy_obs(), deterministic=True)
                    action = int(np.asarray(action).reshape(-1)[0])
                else:
                    keys = pygame.key.get_pressed() if not self.headless else None
                    action = 0
                    if keys:
                        for k, a in HUMAN_KEYS.items():
                            if keys[k]:
                                action = a
                                break

                self._obs, reward, term, trunc, _ = self.env.step(action)
                frame = self.env.render()
                self.episode_return += float(reward)
                self.steps += 1
                total_steps += 1
                if term or trunc:
                    self.best = max(self.best, self.episode_return)
                    self.episodes += 1
                    print(f"episode {self.episodes}: score {self.episode_return:.0f} "
                          f"steps {self.steps} mode={'agent' if self.agent_mode else 'human'}", flush=True)
                    self.reset_episode()

            if not self.headless:
                self.draw(frame)
                self.clock.tick(self.fps)
            else:
                if self.headless and self.episodes == 0 and self.steps == 0:
                    pass
                time.sleep(max(0.0, target_dt - (time.time() - now)))
                if total_steps >= self.headless:
                    if self.shot:
                        out = Path(self.shot)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        self.draw(frame)
                        pygame.image.save(self.screen, str(out))
                        print("screenshot:", out, f"score={self.episode_return:.0f}")
                    running = False

        if not self.headless:
            pygame.quit()
        self.env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/mspacman/dqn_envpool/final_4000000.zip")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--headless", type=int, default=0, help="run N steps with no window (smoke test)")
    p.add_argument("--shot", default=None, help="save a PNG after the headless run")
    args = p.parse_args()

    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"no model at {path}")

    game = PacmanArcade(path, scale=args.scale, seed=args.seed, device=args.device,
                        fps=args.fps, headless=args.headless, shot=args.shot)
    game.run()


if __name__ == "__main__":
    main()
