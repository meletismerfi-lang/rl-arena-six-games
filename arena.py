"""Arena — play the bots, or watch them fight each other.

  .venv/Scripts/python.exe arena.py

PLAY: you against each agent.  WATCH: agent against agent.
Every entry runs in its own process, so a crash in one game never takes the menu down.
UP/DOWN or mouse to pick · ENTER to launch · ESC to quit.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parent

PLAY = [
    ("CHESS", "vs Stockfish", ["chess_ui.py", "--human", "white", "--elo", "1600"]),
    ("FLAPPY BIRD", "vs a solved PPO bot (103 pipes)", ["flappy_duel.py"]),
    ("MS. PAC-MAN", "vs the 2,355 DQN", ["pacman_duel.py"]),
    ("BACKGAMMON", "vs the TD(lambda) net", ["backgammon_ui.py"]),
    ("SCRABBLE", "vs the trie engine", ["scrabble_agent.py", "--play"]),
]

WATCH = [
    ("CHESS — AI vs AI", "2400 (white) vs 3200 (black)", ["chess_ai.py"]),
    ("FLAPPY — AI vs AI", "103-pipe model vs 77-pipe model", ["flappy_duel.py", "--both-bots"]),
    ("PAC-MAN — AI vs AI", "4M model (2,355) vs 10M model (1,317)", ["pacman_ai.py"]),
    ("BACKGAMMON — AI vs AI", "TD net vs pip-count baseline", ["backgammon_ai.py"]),
    ("DOOM — watch", "ViZDoom PPO clearing defend_the_center", ["doom_wsl.py"]),
]

SECTIONS = [("PLAY  ·  you vs the agent", PLAY), ("WATCH  ·  agent vs agent", WATCH)]


def rows():
    """Flat list of (kind, payload) so keyboard navigation can skip section headers."""
    out = []
    for title, items in SECTIONS:
        out.append(("header", title))
        for it in items:
            out.append(("item", it))
    return out


def draw(screen, font, big, small, flat_idx: int) -> None:
    w, h = screen.get_size()
    screen.fill((14, 16, 22))
    title = big.render("RL ARENA", True, (235, 240, 250))
    screen.blit(title, (w // 2 - title.get_width() // 2, 16))
    sub = small.render("five games to play, four to spectate", True, (120, 132, 150))
    screen.blit(sub, (w // 2 - sub.get_width() // 2, 46))

    y = 72
    for i, (kind, payload) in enumerate(rows()):
        if kind == "header":
            screen.blit(small.render(payload, True, (96, 170, 205)), (42, y))
            y += 24
            continue
        name, blurb, _ = payload
        sel = i == flat_idx
        box = pygame.Rect(40, y, w - 80, 54)
        pygame.draw.rect(screen, (44, 58, 82) if sel else (28, 33, 43), box, border_radius=7)
        pygame.draw.rect(screen, (90, 200, 255) if sel else (42, 48, 60), box, 2, border_radius=7)
        screen.blit(font.render(name, True, (240, 245, 250) if sel else (188, 196, 208)), (54, y + 7))
        screen.blit(small.render(blurb, True, (140, 152, 170)), (54, y + 31))
        y += 60

    foot = small.render("UP/DOWN or mouse · ENTER launch · ESC quit", True, (100, 110, 125))
    screen.blit(foot, (w // 2 - foot.get_width() // 2, h - 22))


def launch(cmd: list[str]) -> None:
    path = ROOT / cmd[0]
    if not path.exists() and cmd[0] != "doom_wsl.py":
        print(f"missing {path}")
        return
    print("launching", " ".join(cmd))
    if path.name == "doom_wsl.py":
        # ViZDoom segfaults on this Windows box; it runs in WSL, drawing through WSLg
        cmd = ["wsl.exe", "-d", "jaxgpu", "-u", "root", "--", "bash", "-lc",
               "cd /mnt/c/Users/melet/AI/rl-games && exec /root/gpurl/bin/python doom_play.py "
               "--model models/doom/defend_the_center/ppo_v2/best_model.zip --episodes 5 --live"]
        subprocess.run(cmd, cwd=str(ROOT))
        return
    if path.name == "scrabble_agent.py":
        # typed input needs a real console, which a pygame child does not have
        subprocess.Popen([sys.executable, str(path), *cmd[1:]], cwd=str(ROOT),
                         creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        return
    subprocess.run([sys.executable, str(path), *cmd[1:]], cwd=str(ROOT))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((640, 706))
    pygame.display.set_caption("RL Arena — play the bots, or watch them")
    font = pygame.font.SysFont("consolas", 17, bold=True)
    big = pygame.font.SysFont("consolas", 26, bold=True)
    small = pygame.font.SysFont("consolas", 12)

    entries = rows()
    items = [i for i, (k, _) in enumerate(entries) if k == "item"]
    sel = 0
    guard_until = 0.0        # a closing game hands its click to the arena; ignore that click

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(items)
                elif ev.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(items)
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    draw(screen, font, big, small, items[sel])
                    pygame.display.flip()
                    launch(entries[items[sel]][1][2])
                    pygame.event.clear()
                    guard_until = time.time() + 0.6
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if time.time() < guard_until:
                    continue
                y = 72
                fired = False
                for kind, payload in entries:
                    if kind == "header":
                        y += 24
                        continue
                    if not fired and pygame.Rect(40, y, screen.get_width() - 80, 54).collidepoint(ev.pos):
                        fired = True
                        launch(payload[2])
                        pygame.event.clear()
                        guard_until = time.time() + 0.6
                    y += 60
        draw(screen, font, big, small, items[sel])
        pygame.display.flip()
        pygame.time.wait(16)
    pygame.quit()


if __name__ == "__main__":
    main()
