"""Flappy Bird as a gymnasium env: numpy physics, pygame-rendered frames, Atari-style observations.

Observation: (84, 84, 1) uint8 grayscale (stack 4 outside with VecFrameStack, same as the
Atari pipeline). Actions: Discrete(2) — 0 glide, 1 flap. Reward: +1 per pipe passed, -1 on
death, +0.01 per frame survived (keeps the agent moving instead of hovering at the ceiling).
"""
from __future__ import annotations

import numpy as np
import pygame
from gymnasium import spaces
import gymnasium as gym

W, H = 288, 512          # logical world size (classic flappy proportions)
SCREEN = 84              # observation size
GRAVITY = 0.4
FLAP = -7.0
PIPE_W = 52
GAP = 130
PIPE_SPEED = 3.0
PIPE_SPACING = 190
BIRD_X = 70
BIRD_R = 10


class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, frame_skip: int = 2, max_steps: int = 2000):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.observation_space = spaces.Box(0, 255, (SCREEN, SCREEN, 1), np.uint8)
        self.action_space = spaces.Discrete(2)
        self._surface = None          # created lazily: state-only workers never need SDL
        self.reset()

    # ------------------------------------------------------------------ physics
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bird_y = H * 0.45
        self.vel = 0.0
        self.pipes: list[dict] = []
        self.score = 0
        self.steps = 0
        self.dead = False
        self._spawn_first()
        return self._obs(), {}

    def _spawn_first(self):
        for i in range(3):
            self.pipes.append({"x": W + 120 + i * 190,
                               "gap_y": float(self.np_random.integers(120, H - 120))})

    def _spawn_pipe(self):
        self.pipes.append({"x": self.pipes[-1]["x"] + 190,
                           "gap_y": float(self.np_random.integers(120, H - 120))})

    def step(self, action: int):
        reward = 0.0
        for _ in range(self.frame_skip):
            if int(action) == 1:
                self.vel = FLAP
            self.vel += GRAVITY
            self.bird_y += self.vel

            for pipe in self.pipes:
                pipe["x"] -= PIPE_SPEED
                if not pipe.get("scored") and pipe["x"] + PIPE_W < BIRD_X:
                    pipe["scored"] = True
                    self.score += 1
                    reward += 1.0
            # recycle off-screen pipes
            if self.pipes[0]["x"] < -PIPE_W:
                self.pipes.pop(0)
                self._spawn_pipe()

            if self._collides():
                self.dead = True
                reward = -1.0
                break

        self.steps += 1
        truncated = self.steps >= self.max_steps and not self.dead
        return self._obs(), float(reward), bool(self.dead), bool(truncated), {"score": self.score}

    def _collides(self) -> bool:
        if self.bird_y - BIRD_R < 0 or self.bird_y + BIRD_R > H:
            return True
        for pipe in self.pipes:
            if pipe["x"] < BIRD_X + BIRD_R and pipe["x"] + PIPE_W > BIRD_X - BIRD_R:
                if not (pipe["gap_y"] - GAP / 2 < self.bird_y < pipe["gap_y"] + GAP / 2):
                    return True
        return False

    # ------------------------------------------------------------------ rendering
    def _render_world(self) -> np.ndarray:
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((SCREEN, SCREEN))
        s = self._surface
        sx, sy = SCREEN / W, SCREEN / H
        s.fill((92, 180, 240))                                    # sky
        for pipe in self.pipes:
            top_h = (pipe["gap_y"] - GAP / 2) * sy
            px, pw = pipe["x"] * sx, PIPE_W * sx
            gap_top = (pipe["gap_y"] - GAP / 2) * sy
            gap_bot = (pipe["gap_y"] + GAP / 2) * sy
            pygame.draw.rect(s, (74, 160, 60), (px, 0, pw, top_h))
            pygame.draw.rect(s, (74, 160, 60), (px, gap_bot, pw, SCREEN - gap_bot))
        pygame.draw.rect(s, (222, 216, 149), (0, SCREEN - 4, SCREEN, 4))   # ground
        pygame.draw.circle(s, (250, 215, 60), (int(BIRD_X * sx), int(self.bird_y * sy)), max(2, int(BIRD_R * sy)))
        arr = pygame.surfarray.array3d(s)                          # (SCREEN, SCREEN, 3)
        return np.transpose(arr, (1, 0, 2))                        # (SCREEN, SCREEN, 3)

    def _obs(self) -> np.ndarray:
        rgb = self._render_world()
        return rgb.mean(axis=2).astype(np.uint8)[..., None]

    def render(self):
        return self._render_world() if self.render_mode == "rgb_array" else None

    # ------------------------------------------------------------------ state view
    def next_pipes(self):
        """The two pipes still ahead of the bird, nearest first."""
        ahead = [p for p in self.pipes if p["x"] + PIPE_W > BIRD_X - BIRD_R]
        ahead.sort(key=lambda p: p["x"])
        while len(ahead) < 2:
            ahead.append({"x": (ahead[-1]["x"] + 190) if ahead else W, "gap_y": H / 2})
        return ahead[0], ahead[1]

    def state(self) -> np.ndarray:
        p1, p2 = self.next_pipes()
        return np.array([self.bird_y / H,
                         self.vel / 10.0,
                         (p1["x"] - BIRD_X) / W,
                         (p1["gap_y"] - self.bird_y) / H,
                         (p2["x"] - BIRD_X) / W,
                         (p2["gap_y"] - self.bird_y) / H], dtype=np.float32)


class StateFlappyEnv(FlappyEnv):
    """Same physics and rewards, but a 6-float observation — what timing a flap actually needs.

    At 84x84 the gap is ~21 px and the bird ~3 px, so pixel PPO plateaus at 'survive without
    passing pipes'. The state view trains to indefinite play in a couple of minutes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(-5.0, 5.0, (6,), np.float32)

    def _obs(self) -> np.ndarray:
        return self.state()


if __name__ == "__main__":
    import time
    env = FlappyEnv()
    obs, _ = env.reset(seed=0)
    print("obs", obs.shape, obs.dtype, "| actions", env.action_space.n)
    t0, n, episodes, best = time.time(), 0, 0, 0
    for _ in range(3000):
        obs, r, d, tr, info = env.step(env.action_space.sample())
        n += 1
        if r > 1:
            best = max(best, info["score"])
        if d or tr:
            episodes += 1
            env.reset()
    dt = time.time() - t0
    print(f"{n} steps in {dt:.1f}s = {n/dt:,.0f} steps/s | episodes {episodes} | best random score {best}")
