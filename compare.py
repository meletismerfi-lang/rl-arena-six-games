"""Compare eval curves of several runs on the same game.

  python compare.py pong dqn_cmp ppo_cmp
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def curve(game: str, run: str):
    f = ROOT / "logs" / game / run / "eval" / "evaluations.npz"
    if not f.exists():
        raise SystemExit(f"no eval log at {f}")
    d = np.load(f)
    steps = d["timesteps"]
    means = d["results"].mean(axis=1)
    return steps, means


def main() -> None:
    game, runs = sys.argv[1], sys.argv[2:]
    table = {r: dict(zip(*curve(game, r))) for r in runs}
    all_steps = sorted({s for t in table.values() for s in t})
    print(f"{'steps':>9} " + " ".join(f"{r:>12}" for r in runs))
    for s in all_steps:
        row = " ".join(f"{table[r].get(s, float('nan')):>12.2f}" for r in runs)
        print(f"{s:>9,} {row}")
    print()
    for r in runs:
        st, m = curve(game, r)
        best = m.max()
        print(f"{r:>12}: final={m[-1]:7.2f}  best={best:7.2f} @ {st[m.argmax()]:,} steps")


if __name__ == "__main__":
    main()
