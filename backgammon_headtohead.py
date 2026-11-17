"""Does the 2-ply expectimax actually beat the 1-ply greedy net? Play them against each other.

  .venv/Scripts/python.exe backgammon_headtohead.py --games 200

Both sides share the same value net, so the only difference is the search depth. Colours
alternate every game. This is the honest test of whether the upgrade earned its compute.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch as th

from backgammon_agent import Board, ValueNet, MODEL, WHITE, BLACK, choose


def play(net, device, side_2ply: int, rng) -> int:
    board = Board()
    side = WHITE
    for _ in range(800):
        w = board.winner()
        if w is not None:
            return w
        d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        pips = [d1, d2] if d1 != d2 else [d1] * 4
        if side == side_2ply:
            from backgammon_agent import choose_2ply
            board = choose_2ply(board, side, pips, net, device)
        else:
            board = choose(board, side, pips, net, device)
        side = -side
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = th.load(MODEL, map_location=args.device, weights_only=False)
    net = ValueNet(ck["hidden"]).to(args.device)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    rng = np.random.default_rng(7)
    w2 = l2 = d = 0
    t0 = time.time()
    for g in range(args.games):
        side2 = WHITE if g % 2 == 0 else BLACK
        winner = play(net, args.device, side2, rng)
        if winner == 0:
            d += 1
        elif winner == side2:
            w2 += 1
        else:
            l2 += 1
        if (g + 1) % 25 == 0:
            print(f"  {g+1:3d} games: 2-ply {w2}W / 1-ply {l2}W / {d}D "
                  f"({time.time()-t0:.0f}s)", flush=True)

    n = max(w2 + l2 + d, 1)
    print(f"2-ply {w2}W · 1-ply {l2}W · {d}D over {args.games} games → 2-ply win rate {100*w2/n:.1f}%")


if __name__ == "__main__":
    main()
