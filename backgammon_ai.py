"""Spectate two backgammon agents: the TD(λ) net against a pip-count baseline.

  .venv/Scripts/python.exe backgammon_ai.py           # auto-rolls, plays continuously
  .venv/Scripts/python.exe backgammon_ai.py --mode netvnet

The pip baseline plays the legal move that minimises its own pips — crude, but it is a
genuine strategic opponent, and the TD net beats it ~99% of the time, so expect a rout.
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame
import torch as th

from backgammon_agent import Board, WHITE, ValueNet, choose, choose_2ply, MODEL
from backgammon_ui import (BOARD_W, BOARD_H, HUD, MARGIN, ROWS_H, render as draw_board)


def load_net(device: str):
    ck = th.load(MODEL, map_location=device, weights_only=False)
    net = ValueNet(ck["hidden"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def pip_choice(board: Board, side: int, pips):
    cands = board.legal_moves(side, pips)
    if len(cands) == 1:
        return cands[0]
    return min(cands, key=lambda b: b.pip_count(side) - 0.5 * (15 - (b.off_w if side == WHITE else b.off_b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="netvspip", choices=["netvspip", "netvnet"])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--search", default="2ply", choices=["1ply", "2ply"])
    args = ap.parse_args()

    net_a = load_net(args.device)
    net_b = load_net(args.device) if args.mode == "netvnet" else None
    rng = np.random.default_rng(7)

    pygame.init()
    screen = pygame.display.set_mode((BOARD_W, MARGIN * 2 + ROWS_H * 2 + HUD))
    pygame.display.set_caption("Backgammon — " + ("net vs net" if net_b else "net vs pip baseline"))
    font = pygame.font.SysFont("consolas", 16, bold=True)
    small = pygame.font.SysFont("consolas", 12)
    clock = pygame.time.Clock()

    board = Board()
    side = WHITE
    rounds = {"A": 0, "B": 0}
    history: list[str] = []
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False

        d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        pips = [d1, d2] if d1 != d2 else [d1] * 4
        pick = choose_2ply if args.search == "2ply" else choose
        if side == WHITE:
            board = pick(board, side, pips, net_a, args.device)
            mover = "A (TD net)"
        elif net_b is not None:
            board = pick(board, side, pips, net_b, args.device)
            mover = "B (TD net)"
        else:
            board = pip_choice(board, side, pips)
            mover = "B (pip)"
        history.append(f"{mover} rolled {pips}")
        history = history[-3:]

        winner = board.winner()
        if winner is not None:
            rounds["A" if winner == WHITE else "B"] += 1
            history.append(f"game to {'A' if winner == WHITE else 'B'}")
            board = Board()
            side = WHITE
        else:
            side = -side

        draw_board(screen, board, font, small, pips, None, history[-1] if history else "", True,
                   side, {"you": rounds["A"], "bot": rounds["B"]})
        screen.blit(small.render(f"score  A {rounds['A']} · B {rounds['B']}   "
                                 f"({'net vs net' if net_b else 'net vs pip'}) · ESC quit",
                                 True, (200, 210, 225)), (MARGIN, BOARD_H - 14))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
