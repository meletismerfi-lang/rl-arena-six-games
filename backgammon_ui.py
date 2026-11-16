"""Backgammon: play the TD(λ) agent on a real board.

  .venv/Scripts/python.exe backgammon_ui.py

Click a checker, then a destination (legal destinations light up). Dice are used one at a time,
so you can play a 2- or 4-move sequence in any order. SPACE rolls, ESC quits.

Visuals are drawn procedurally — rimmed checkers, pip dice, wood frame — so there are no image
assets to ship and everything stays crisp at any window size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import pygame.gfxdraw as gfx
import torch as th

from backgammon_agent import Board, WHITE, BLACK, ValueNet, choose, MODEL

ROOT = Path(__file__).resolve().parent
MARGIN, BAR_W, TRAY_W = 30, 46, 46
COL_W = 52
ROWS_H = 268
BOARD_W = MARGIN * 2 + COL_W * 12 + BAR_W + TRAY_W
BOARD_H = MARGIN * 2 + ROWS_H * 2 + 40
HUD = 96

FRAME = (86, 42, 32)
WOOD = (198, 148, 96)
POINT_A = (176, 58, 44)
POINT_B = (236, 220, 194)
CHECK_W, CHECK_W_RIM = (240, 228, 208), (150, 116, 84)
CHECK_B, CHECK_B_RIM = (40, 36, 34), (104, 98, 92)
HILITE = (120, 210, 255)
LEGAL = (120, 220, 160)
INK = (238, 236, 230)


def point_geometry(i: int):
    """(x_center, base_y, apex_y, ydir) — checkers stack from the wide edge inward."""
    if i < 12:                                   # bottom row: wide edge on the bottom rail
        col = 11 - i
        x = MARGIN + COL_W * col + COL_W // 2
        base = MARGIN + ROWS_H * 2
        return x, base, base - 40, -1
    col = i - 12                                 # top row: wide edge on the top rail
    x = MARGIN + COL_W * col + COL_W // 2
    base = MARGIN
    return x, base, base + 40, 1


def bar_x() -> int:
    return MARGIN + COL_W * 12 + BAR_W // 2


def tray_x() -> int:
    return MARGIN + COL_W * 12 + BAR_W + TRAY_W // 2


def draw_checker(surf, cx: int, cy: int, r: int, white: bool) -> None:
    fill, rim = (CHECK_W, CHECK_W_RIM) if white else (CHECK_B, CHECK_B_RIM)
    gfx.filled_circle(surf, cx, cy, r, rim)
    gfx.aacircle(surf, cx, cy, r, rim)
    gfx.filled_circle(surf, cx, cy, max(1, r - 3), fill)
    gfx.aacircle(surf, cx, cy, max(1, r - 3), fill)
    hi = (min(255, fill[0] + 22), min(255, fill[1] + 22), min(255, fill[2] + 22))
    gfx.aacircle(surf, cx, cy, max(1, r - 6), hi)


PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def draw_die(surf, x: int, y: int, size: int, value: int) -> None:
    body = pygame.Rect(x, y, size, size)
    pygame.draw.rect(surf, (250, 248, 244), body, border_radius=6)
    pygame.draw.rect(surf, (40, 38, 36), body, 2, border_radius=6)
    step = size // 4
    cx, cy = x + size // 2, y + size // 2
    for dx, dy in PIPS[value]:
        px, py = cx + dx * step, cy + dy * step
        gfx.filled_circle(surf, px, py, max(2, size // 9), (36, 34, 32))
        gfx.aacircle(surf, px, py, max(2, size // 9), (36, 34, 32))


def legal_targets(board: Board, side: int, pips, selected) -> set[int]:
    """Which point indices (or -1 for the tray) the held checker can reach with any die."""
    if selected is None or not pips:
        return set()
    out = set()
    for die in set(pips):
        for dest, _hit in board._destinations(board, side, die):
            if board._source_point(board, side, die, dest) == selected:
                out.add(-1 if dest == -1 else dest)
    return out


def render(screen, board: Board, font, small, pips, selected, message, human_white: bool,
           side: int = WHITE, score: dict | None = None) -> None:
    screen.fill((22, 20, 20))
    frame = pygame.Rect(MARGIN - 10, MARGIN - 10, COL_W * 12 + BAR_W + TRAY_W + 20, ROWS_H * 2 + 20)
    pygame.draw.rect(screen, FRAME, frame, border_radius=10)
    play = pygame.Rect(MARGIN, MARGIN, COL_W * 12 + BAR_W + TRAY_W, ROWS_H * 2)
    pygame.draw.rect(screen, WOOD, play)
    pygame.draw.rect(screen, (120, 78, 54), (MARGIN + COL_W * 12, MARGIN, BAR_W, ROWS_H * 2))

    targets = legal_targets(board, side, pips, selected)

    for i in range(24):
        x, base, apex, ydir = point_geometry(i)
        color = POINT_A if (i % 2) else POINT_B
        poly = [(x - COL_W // 2 + 4, base), (x + COL_W // 2 - 4, base), (x, apex)]
        gfx.filled_polygon(screen, poly, color)
        gfx.aapolygon(screen, poly, (60, 40, 30))
        if selected == i:
            pygame.draw.polygon(screen, HILITE, poly, 3)
        if i in targets:
            gfx.filled_circle(screen, x, base + ydir * 26, 7, LEGAL)
            gfx.aacircle(screen, x, base + ydir * 26, 7, LEGAL)

        n = int(board.points[i])
        count = min(abs(n), 8)
        white = n > 0
        for k in range(count):
            cy = base + ydir * (15 + k * 30)
            draw_checker(screen, x, cy, 14, white)
        if abs(n) > count:
            screen.blit(small.render(f"x{abs(n)}", True, INK if white else (250, 230, 60)),
                        (x - 8, base + ydir * (15 + count * 30)))

    for k in range(min(board.bar_w, 6)):
        draw_checker(screen, bar_x(), MARGIN + 22 + k * 30, 14, True)
    for k in range(min(board.bar_b, 6)):
        draw_checker(screen, bar_x(), MARGIN + ROWS_H * 2 - 22 - k * 30, 14, False)
    if board.bar_w:
        screen.blit(small.render(str(board.bar_w), True, CHECK_W_RIM), (bar_x() + 16, MARGIN + 10))
    if board.bar_b:
        screen.blit(small.render(str(board.bar_b), True, (210, 210, 210)),
                    (bar_x() + 16, MARGIN + ROWS_H * 2 - 30))

    tx = tray_x()
    pygame.draw.rect(screen, (56, 44, 38), (tx - 16, MARGIN, 32, ROWS_H * 2), border_radius=4)
    if -1 in targets:
        pygame.draw.rect(screen, LEGAL, (tx - 16, MARGIN, 32, ROWS_H * 2), 3, border_radius=4)
    screen.blit(font.render(str(board.off_w), True, CHECK_W), (tx - 8, MARGIN + ROWS_H - 60))
    screen.blit(font.render(str(board.off_b), True, (190, 190, 190)), (tx - 8, MARGIN + ROWS_H + 30))

    y = MARGIN + ROWS_H * 2 + 14
    if pips:
        for i, v in enumerate(pips[:4]):
            draw_die(screen, MARGIN + i * 42, y, 34, v)
    else:
        screen.blit(small.render("no dice — SPACE to roll", True, (140, 140, 150)), (MARGIN, y + 10))
    screen.blit(font.render(message, True, (150, 200, 240)), (MARGIN + 190, y + 2))
    screen.blit(small.render("click checker then destination · SPACE roll · ESC quit",
                             True, (120, 130, 145)), (MARGIN + 190, y + 26))
    who = "WHITE" if human_white else "BLACK"
    screen.blit(small.render(f"you: {who}   to move: {'WHITE' if side == WHITE else 'BLACK'}   "
                             f"pips W {board.pip_count(WHITE)} / B {board.pip_count(BLACK)}",
                             True, (150, 156, 168)), (MARGIN, y + 48))
    if score:
        screen.blit(small.render(f"score  you {score.get('you', 0)} · bot {score.get('bot', 0)}",
                                 True, (200, 210, 225)), (BOARD_W - 210, y + 48))


def point_at(pos) -> int | None:
    x, y = pos
    if not (MARGIN <= x <= MARGIN + COL_W * 12 and MARGIN <= y <= MARGIN + ROWS_H * 2):
        return None
    col = (x - MARGIN) // COL_W
    if y < MARGIN + ROWS_H:
        return int(12 + col)
    return int(11 - col)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="white", choices=["white", "black"])
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    net = ValueNet().to(device)
    if MODEL.exists():
        ck = th.load(MODEL, map_location=device, weights_only=False)
        net = ValueNet(ck["hidden"]).to(device)
        net.load_state_dict(ck["state_dict"])
    net.eval()

    human_side = WHITE if args.human == "white" else BLACK
    rng = np.random.default_rng()
    board = Board()
    side = WHITE
    pips: list[int] = []
    selected = None
    message = "your turn — SPACE to roll"
    rounds = {"you": 0, "bot": 0}

    pygame.init()
    screen = pygame.display.set_mode((BOARD_W, MARGIN * 2 + ROWS_H * 2 + HUD))
    pygame.display.set_caption("Backgammon — you vs the TD(lambda) agent")
    font = pygame.font.SysFont("consolas", 17, bold=True)
    small = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()

    def start_turn() -> None:
        nonlocal pips, side, message
        d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        pips = [d1, d2] if d1 != d2 else [d1] * 4
        message = f"your move — dice {pips}" if side == human_side else f"bot thinking — dice {pips}"

    def end_turn() -> None:
        nonlocal side, board, message
        winner = board.winner()
        if winner:
            rounds["you" if winner == human_side else "bot"] += 1
            board = Board()
            side = WHITE
            message = "new game — SPACE to roll"
        else:
            side = -side
            message = "your turn — SPACE to roll" if side == human_side else "bot's turn"

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE and not pips:
                    start_turn()
            elif ev.type == pygame.MOUSEBUTTONDOWN and pips and side == human_side:
                idx = point_at(ev.pos)
                if idx is None:
                    continue
                n = int(board.points[idx])
                if selected is None:
                    if (n > 0) if human_side == WHITE else (n < 0):
                        selected = idx
                        message = "now click a lit destination"
                else:
                    die = h = target = None
                    for d in sorted(set(pips)):
                        for dest, hit in board._destinations(board, human_side, d):
                            if board._source_point(board, human_side, d, dest) == selected and dest == idx:
                                die, target, h = d, dest, hit
                                break
                        if die:
                            break
                    if die:
                        board._apply(board, human_side, selected, target, h)
                        pips.remove(die)
                        selected = None
                        message = f"played — dice left {pips}" if pips else "turn over"
                        if not pips:
                            end_turn()
                    else:
                        selected = None
                        message = "illegal — pick another checker"

        if pips and side != human_side:
            board = choose(board, side, pips, net, device)
            pips = []
            end_turn()

        render(screen, board, font, small, pips, selected, message, human_side == WHITE, side, rounds)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
