"""Spectate two chess engines playing each other, drawn with the same board as chess_ui.

  .venv/Scripts/python.exe chess_ai.py                      # 2400 vs 3400
  .venv/Scripts/python.exe chess_ai.py --elo-a 2800 --elo-b 1320 --movetime 0.3

Useful as a sanity check that the harness plays legal, terminating games at both ends of the
strength dial. The score line counts finished games.
"""
from __future__ import annotations

import argparse
import time

import chess
import pygame

from chess_agent import ChessBot
from chess_ui import (BOARD_PX, SQ, HUD, draw_board)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo-a", type=int, default=2400)
    ap.add_argument("--elo-b", type=int, default=3200)
    ap.add_argument("--movetime", type=float, default=0.4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--plies-per-second", type=float, default=1.2, help="animation speed")
    args = ap.parse_args()

    bot_a = ChessBot(elo=args.elo_a, movetime=args.movetime)
    bot_b = ChessBot(elo=args.elo_b, movetime=args.movetime)

    pygame.init()
    screen = pygame.display.set_mode((BOARD_PX, BOARD_PX + HUD))
    pygame.display.set_caption("Chess — bot vs bot")
    font_piece = pygame.font.SysFont("segoeuisymbol,arial", int(SQ * 0.62), bold=True)
    font_hud = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    board = chess.Board()
    last = None
    score = {"A": 0, "B": 0, "draw": 0}
    next_move_at = time.time()
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False

        if not board.is_game_over() and time.time() >= next_move_at:
            bot = bot_a if board.turn == chess.WHITE else bot_b
            mv = bot.move(board)
            board.push(mv)
            last = mv
            next_move_at = time.time() + 1.0 / max(args.plies_per_second, 0.1)
        elif board.is_game_over():
            res = board.result()
            if res == "1-0":
                score["A"] += 1
            elif res == "0-1":
                score["B"] += 1
            else:
                score["draw"] += 1
            print(f"game over {res} ({board.outcome().termination.name}) -> {score}")
            board = chess.Board()
            last = None
            next_move_at = time.time() + 1.0

        status = (f"A({args.elo_a}) vs B({args.elo_b}) · W {score['A']} / D {score['draw']} / B {score['B']}"
                  if not board.is_game_over() else f"result {board.result()}")
        draw_board(screen, board, font_piece, font_hud, None, last, status, 0)
        pygame.display.flip()
        clock.tick(args.fps)

    bot_a.close()
    bot_b.close()
    pygame.quit()


if __name__ == "__main__":
    main()
