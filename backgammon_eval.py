"""How strong is the TD(λ) backgammon net? Play it against a random-move opponent.

  .venv/Scripts/python.exe backgammon_eval.py --games 200

A random mover is a weak baseline but a meaningful floor: it plays legal backgammon, hits
blots when they are there, and never resigns. Win rate above ~85% means the net has learned
real race/holding structure rather than just "roll and move".
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch as th

from backgammon_agent import Board, ValueNet, MODEL, WHITE, BLACK, choose


def pip_move(board: Board, side: int, pips, rng):
    """Baseline opponent: play the move that minimises its own pip count."""
    cands = board.legal_moves(side, pips)
    if len(cands) == 1:
        return cands[0]
    best = min(cands, key=lambda b: b.pip_count(side) - 0.5 * (15 - (b.off_w if side == WHITE else b.off_b)))
    return best


def play_one(net, device, net_side: int, rng, baseline: str = "random") -> int:
    """Return the winning side (+1 white / -1 black)."""
    board = Board()
    side = WHITE
    for _ in range(600):
        w = board.winner()
        if w is not None:
            return w
        d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        pips = [d1, d2] if d1 != d2 else [d1] * 4
        if side == net_side:
            board = choose(board, side, pips, net, device)
        elif baseline == "pip":
            board = pip_move(board, side, pips, rng)
        else:
            cands = board.legal_moves(side, pips)
            board = cands[int(rng.integers(len(cands)))]
        side = -side
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--baseline", default="random", choices=["random", "pip"])
    args = ap.parse_args()

    ck = th.load(MODEL, map_location=args.device, weights_only=False)
    net = ValueNet(ck["hidden"]).to(args.device)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    rng = np.random.default_rng(2024)
    wins = losses = draws = 0
    t0 = time.time()
    for g in range(args.games):
        net_side = WHITE if g % 2 == 0 else BLACK          # alternate colour, fair
        winner = play_one(net, args.device, net_side, rng, args.baseline)
        if winner == 0:
            draws += 1
        elif winner == net_side:
            wins += 1
        else:
            losses += 1
    n = max(wins + losses + draws, 1)
    print(f"{args.games} games in {time.time()-t0:.0f}s — net {wins}W / {losses}L / {draws}D "
          f"= {100*wins/n:.1f}% win rate vs {args.baseline}")


if __name__ == "__main__":
    main()
