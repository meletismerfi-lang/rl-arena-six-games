"""Play Flappy Bird against the bot, head to head, in one window.

  .venv/Scripts/python.exe flappy_duel.py            # SPACE to flap, race the bot
  .venv/Scripts/python.exe flappy_duel.py --bot-pipes 0

Both birds fly through the same pipe field and the same gaps, so it is a straight race.
First to fall loses that round; score resets on R. The bot reads the same 6-float state the
policy was trained on, so it never cheats by seeing the future.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch as th
from stable_baselines3 import PPO

from flappy_env import FlappyEnv, W, H, PIPE_W, GAP, PIPE_SPEED, PIPE_SPACING, GRAVITY, FLAP, BIRD_X, BIRD_R

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "flappy" / "ppo_state" / "final_long.zip"
MODEL_B = ROOT / "models" / "flappy" / "ppo_state" / "final.zip"


class Bird:
    def __init__(self, y: float):
        self.y = y
        self.vel = 0.0
        self.dead = False
        self.score = 0
        self.color = (250, 215, 60)


class Duel:
    def __init__(self, bot_handicap_pipes: int = 0, both_bots: bool = False, model_b: Path | None = None):
        self.env = FlappyEnv()          # pipe field + physics source of truth
        self.model = None
        self.both_bots = both_bots
        if MODEL.exists():
            self.model = PPO.load(str(MODEL), device="cpu")
        self.model_b = None
        if both_bots:
            mb = model_b or MODEL_B
            if mb.exists():
                self.model_b = PPO.load(str(mb), device="cpu")
        self.handicap = bot_handicap_pipes
        self.reset()

    def reset(self):
        self.env.reset()
        self.human = Bird(H * 0.42)
        self.bot = Bird(H * 0.58)
        self.human.color = (110, 220, 255)
        self.bot.color = (255, 140, 60)
        self.rounds = {"human": 0, "bot": 0, "tie": 0}

    def _state_for(self, bird: Bird) -> np.ndarray:
        p1, p2 = self.env.next_pipes() if hasattr(self.env, "next_pipes") else (self.env.pipes[0], self.env.pipes[1])
        return np.array([bird.y / H, bird.vel / 10.0,
                         (p1["x"] - BIRD_X) / W, (p1["gap_y"] - bird.y) / H,
                         (p2["x"] - BIRD_X) / W, (p2["gap_y"] - bird.y) / H], dtype=np.float32)

    def _advance(self, bird: Bird, flap: bool) -> None:
        if bird.dead:
            return
        for _ in range(self.env.frame_skip):
            if flap:
                bird.vel = FLAP
            bird.vel += GRAVITY
            bird.y += bird.vel
            if bird.y - BIRD_R < 0 or bird.y + BIRD_R > H:
                bird.dead = True
                return
            for pipe in self.env.pipes:
                if pipe["x"] < BIRD_X + BIRD_R and pipe["x"] + PIPE_W > BIRD_X - BIRD_R:
                    if not (pipe["gap_y"] - GAP / 2 < bird.y < pipe["gap_y"] + GAP / 2):
                        bird.dead = True
                        return
        for pipe in self.env.pipes:
            if not pipe.get(f"scored_{id(bird)}") and pipe["x"] + PIPE_W < BIRD_X:
                pipe[f"scored_{id(bird)}"] = True
                bird.score += 1

    def _bot_flap(self, bird: Bird, model) -> bool:
        if model is None or bird.dead:
            return False
        act, _ = model.predict(self._state_for(bird), deterministic=True)
        return bool(int(act))

    def step(self, human_flap: bool) -> None:
        if self.both_bots:
            self._advance(self.human, self._bot_flap(self.human, self.model_b))
            self._advance(self.bot, self._bot_flap(self.bot, self.model))
            self._scroll()
            self._check_round()
            return
        self._advance(self.human, human_flap)
        bot_flap = False
        if self.model is not None and not self.bot.dead and self.bot.score >= self.handicap:
            act, _ = self.model.predict(self._state_for(self.bot), deterministic=True)
            bot_flap = bool(int(act))
        elif not self.bot.dead:
            # handicap: fly a simple hold-height policy before the net takes over
            p1, _ = (self.env.next_pipes() if hasattr(self.env, "next_pipes") else (self.env.pipes[0], self.env.pipes[1]))
            bot_flap = self.bot.y > p1["gap_y"] + 8 or self.bot.vel > 3
        self._advance(self.bot, bot_flap)
        self._scroll()
        self._check_round()

    def _scroll(self) -> None:
        for pipe in self.env.pipes:
            pipe["x"] -= PIPE_SPEED
        if self.env.pipes[0]["x"] < -PIPE_W:
            self.env.pipes.pop(0)
            self.env.pipes.append({"x": self.env.pipes[-1]["x"] + PIPE_SPACING,
                                   "gap_y": float(self.env.np_random.integers(120, H - 120))})
    def _check_round(self) -> None:
        if self.human.dead or self.bot.dead:
            h, b = self.human.score, self.bot.score
            if h == b:
                self.rounds["tie"] += 1
            elif not self.human.dead or h > b:
                self.rounds["human"] += 1
            else:
                self.rounds["bot"] += 1
            self.reset()


def render(screen, duel: Duel, font, big) -> None:
    scale = screen.get_width() / W
    surf = pygame.Surface((W, H))
    surf.fill((92, 180, 240))
    for pipe in duel.env.pipes:
        pygame.draw.rect(surf, (74, 160, 60), (pipe["x"], 0, PIPE_W, pipe["gap_y"] - GAP / 2))
        pygame.draw.rect(surf, (74, 160, 60), (pipe["x"], pipe["gap_y"] + GAP / 2, PIPE_W, H))
    pygame.draw.rect(surf, (222, 216, 149), (0, H - 24, W, 24))
    for bird in (duel.human, duel.bot):
        if not bird.dead:
            pygame.draw.circle(surf, bird.color, (BIRD_X, int(bird.y)), BIRD_R)
            pygame.draw.circle(surf, (20, 20, 25), (BIRD_X + 4, int(bird.y) - 2), 2)
    screen.blit(pygame.transform.scale(surf, (int(W * scale), int(H * scale))), (0, 0))
    y = int(H * scale) + 6
    left = "BOT-A" if duel.both_bots else "YOU"
    screen.blit(big.render(f"{left} {duel.human.score:>3}    BOT {duel.bot.score:>3}", True, (240, 240, 245)), (10, y))
    screen.blit(font.render(f"rounds — you {duel.rounds['human']}  bot {duel.rounds['bot']}  tie {duel.rounds['tie']}",
                            True, (150, 160, 175)), (10, y + 26))
    screen.blit(font.render("SPACE flap · R reset round · ESC quit", True, (120, 130, 145)), (10, y + 46))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.15)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bot-pipes", type=int, default=0, help="bot starts this many pipes behind")
    ap.add_argument("--both-bots", action="store_true", help="spectate: two different bots race")
    ap.add_argument("--model-b", default=None, help="left-hand bot when --both-bots")
    args = ap.parse_args()

    duel = Duel(args.bot_pipes, both_bots=args.both_bots,
                model_b=Path(args.model_b) if args.model_b else None)
    pygame.init()
    size = (int(W * args.scale), int(H * args.scale) + 78)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Flappy Bird — bot vs bot" if args.both_bots else "Flappy Bird — you vs the bot")
    font = pygame.font.SysFont("consolas", 14)
    big = pygame.font.SysFont("consolas", 20, bold=True)
    clock = pygame.time.Clock()

    running = True
    while running:
        flap = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    flap = True
                elif ev.key == pygame.K_r:
                    duel.reset()
        duel.step(flap)
        render(screen, duel, font, big)
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
