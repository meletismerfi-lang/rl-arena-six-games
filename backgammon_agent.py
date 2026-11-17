"""Backgammon: full rules, TD(λ) self-play learning, club-strength play.

Board: `points[0..23]` signed — positive = White checkers, negative = Black.
White moves 23 -> 0 (home 0..5, bears off below 0); Black moves 0 -> 23 (home 18..23).
White = +1 side, Black = -1 side; the network always sees the board from the mover's view.

Network: 198 inputs (TD-Gammon encoding), 2x256 hidden, tanh output in (-1, 1) = expected
outcome for the side to move. Learning: TD(λ) with eligibility traces over self-play games,
greedy 1-ply move selection (evaluate every legal move sequence after the dice).

  .venv/Scripts/python.exe backgammon_agent.py --train 40000
  .venv/Scripts/python.exe backgammon_agent.py --eval 500
  .venv/Scripts/python.exe backgammon_agent.py --play --human white
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "backgammon" / "td_agent.pt"

WHITE, BLACK = 1, -1

START = np.zeros(24, dtype=np.int8)
START[23] = 2      # white (moves 23->0)
START[12] = 5
START[7] = 3
START[5] = 5
START[0] = -2      # black (moves 0->23)
START[11] = -5
START[16] = -3
START[18] = -5


class Board:
    """Mutable backgammon position for one side."""

    __slots__ = ("points", "bar_w", "bar_b", "off_w", "off_b")

    def __init__(self, points=None, bar_w=0, bar_b=0, off_w=0, off_b=0):
        self.points = START.copy() if points is None else points.astype(np.int8).copy()
        self.bar_w, self.bar_b, self.off_w, self.off_b = bar_w, bar_b, off_w, off_b

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.points = self.points.copy()
        b.bar_w, b.bar_b, b.off_w, b.off_b = self.bar_w, self.bar_b, self.off_w, self.off_b
        return b

    # ---------------------------------------------------------------- helpers
    def home(self, side: int) -> range:
        return range(0, 6) if side == WHITE else range(18, 24)

    def all_home(self, side: int) -> bool:
        """True when `side` may bear off: no checkers on the bar or outside its home board."""
        if side == WHITE:
            if self.bar_w:
                return False
            return all(self.points[i] <= 0 for i in range(6, 24))
        if self.bar_b:
            return False
        return all(self.points[i] >= 0 for i in range(0, 18))

    def pip_count(self, side: int) -> int:
        """Steps remaining to bear everything off."""
        if side == WHITE:
            return int(sum(int(v) * (i + 1) for i, v in enumerate(self.points) if v > 0) + self.bar_w * 25)
        return int(sum(-int(v) * (24 - i) for i, v in enumerate(self.points) if v < 0) + self.bar_b * 25)

    def winner(self) -> int | None:
        if self.off_w >= 15:
            return WHITE
        if self.off_b >= 15:
            return BLACK
        return None

    # ---------------------------------------------------------------- encoding
    def encode(self, side: int) -> np.ndarray:
        """198 inputs from `side`'s perspective (opponent on the far side of the encoding)."""
        me = self.points if side == WHITE else -self.points[::-1]
        opp = -me
        x = np.zeros(198, dtype=np.float32)
        k = 0
        for arr in (me, opp):
            for i in range(24):
                n = int(arr[i])
                if n <= 0:
                    x[k:k + 3] = 0.0
                else:
                    x[k:k + 3] = [1.0 if n >= j else 0.0 for j in (1, 2, 3)]
                k += 3
        # counts above 3 as a 4th channel, plus bar / off
        for arr in (me, opp):
            for i in range(24):
                x[k] = max(0.0, float(arr[i]) - 3) / 2.0
                k += 1
        my_bar, op_bar = (self.bar_w, self.bar_b) if side == WHITE else (self.bar_b, self.bar_w)
        my_off, op_off = (self.off_w, self.off_b) if side == WHITE else (self.off_b, self.off_w)
        x[k:k + 6] = [my_bar / 2, op_bar / 2, my_off / 15, op_off / 15, 0.0, 0.0]
        return x

    # ---------------------------------------------------------------- moves
    def legal_moves(self, side: int, pips: list[int]) -> list["Board"]:
        """Distinct resulting boards using the maximum number of available pips.

        `pips` is the individual die values — 2 normally, 4 for doubles — so the
        standard "must play as many dice as possible" rule falls out of the search.
        """
        found: list[tuple[int, Board]] = []
        seen: set[bytes] = set()

        def rec(b: Board, remaining: list[int], used: int) -> None:
            if not remaining:
                found.append((used, b))
                return
            progressed = False
            for die in sorted(set(remaining)):
                dests = self._destinations(b, side, die)
                if not dests:
                    continue
                progressed = True
                rest = remaining.copy()
                rest.remove(die)
                for dest, hit in dests:
                    nb = b.copy()
                    src = self._source_point(b, side, die, dest)
                    self._apply(nb, side, src, dest, hit)
                    rec(nb, rest, used + 1)
            if not progressed:
                found.append((used, b))

        rec(self, list(pips), 0)
        if not found:
            return [self.copy()]
        best = max(u for u, _ in found)
        out: list[Board] = []
        for u, m in found:
            if u < best:
                continue
            key = m.points.tobytes() + bytes([m.bar_w, m.bar_b, m.off_w, m.off_b])
            if key not in seen:
                seen.add(key)
                out.append(m)
        return out or [self.copy()]

    def _source_point(self, b: Board, side: int, die: int, dest: int) -> int:
        if dest == -1:
            # bearing off: the source is the highest (white) / lowest (black) checker that can go
            if side == WHITE:
                cand = [i for i in range(6) if b.points[i] > 0 and (i - die < 0)]
                exact = [i for i in cand if i - die == -1]
                return exact[0] if exact else max(cand)
            cand = [i for i in range(18, 24) if b.points[i] < 0 and (i + die > 23)]
            exact = [i for i in cand if i + die == 24]
            return exact[0] if exact else min(cand)
        if side == WHITE:
            return dest + die if dest + die <= 23 else -1
        return dest - die if dest - die >= 0 else -1

    def _destinations(self, b: Board, side: int, die: int) -> list[tuple[int, bool]]:
        """Return (dest, hits) candidates for one die; dest == -1 means bear off."""
        out: list[tuple[int, bool]] = []
        bar = b.bar_w if side == WHITE else b.bar_b
        if bar:
            dest = 24 - die if side == WHITE else die - 1
            if self._can_land(b, side, dest):
                out.append((dest, self._hits(b, side, dest)))
            return out

        for i in range(24):
            if (b.points[i] > 0) if side == WHITE else (b.points[i] < 0):
                dest = i - die if side == WHITE else i + die
                if 0 <= dest <= 23:
                    if self._can_land(b, side, dest):
                        out.append((dest, self._hits(b, side, dest)))
                else:
                    # bearing off
                    if b.all_home(side):
                        home = b.home(side)
                        if dest == (-1 if side == WHITE else 24):
                            out.append((-1, False))
                        else:
                            # overshoot allowed only for the farthest checker
                            occupied = [p for p in home if (b.points[p] > 0 if side == WHITE else b.points[p] < 0)]
                            farthest = min(occupied) if side == WHITE else max(occupied)
                            if i == farthest:
                                out.append((-1, False))
        return out

    @staticmethod
    def _can_land(b: Board, side: int, dest: int) -> bool:
        if side == WHITE:
            return b.points[dest] > -2
        return b.points[dest] < 2

    @staticmethod
    def _hits(b: Board, side: int, dest: int) -> bool:
        if side == WHITE:
            return b.points[dest] == -1
        return b.points[dest] == 1

    @staticmethod
    def _apply(b: Board, side: int, src: int, dest: int, hit: bool) -> None:
        if src >= 0:
            b.points[src] -= side
        else:
            if side == WHITE:
                b.bar_w -= 1
            else:
                b.bar_b -= 1
        if dest == -1:
            if side == WHITE:
                b.off_w += 1
            else:
                b.off_b += 1
            return
        if hit:
            if side == WHITE:
                b.points[dest] = 0
                b.bar_b += 1
            else:
                b.points[dest] = 0
                b.bar_w += 1
        b.points[dest] += side


class ValueNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(198, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def choose(board: Board, side: int, pips, net, device) -> Board:
    cands = board.legal_moves(side, list(pips))
    if len(cands) == 1:
        return cands[0]
    x = th.from_numpy(np.stack([m.encode(side) for m in cands])).to(device)
    with th.no_grad():
        v = net(x)
    return cands[int(th.argmax(v).item())]


DICE21 = [(a, b) for a in range(1, 7) for b in range(a, 7)]


def choose_2ply(board: "Board", side: int, pips, net, device, top_k: int = 6) -> "Board":
    """Expectimax over the opponent's next roll — a real strength jump over 1-ply greed.

    Rank the legal moves with the value net (1-ply), keep the best `top_k`, then for each of
    those play out every one of the 21 distinct dice rolls and let the opponent take their best
    reply by the same net. The move's value is the mean of -V(reply) across rolls, because the
    net always reports the expectation of the side to move.
    """
    cands = board.legal_moves(side, pips)
    if len(cands) <= 1:
        return cands[0] if cands else board.copy()

    with th.no_grad():
        v1 = net(th.from_numpy(np.stack([c.encode(side) for c in cands])).to(device))
    order = th.argsort(v1, descending=True)[:top_k].tolist()
    finalists = [cands[i] for i in order]
    if len(finalists) == 1:
        return finalists[0]

    best_value, best_board = -9e9, finalists[0]
    for cand in finalists:
        total = 0.0
        for d1, d2 in DICE21:
            d_pips = [d1, d2] if d1 != d2 else [d1] * 4
            replies = cand.legal_moves(-side, d_pips)
            if not replies:
                total += 0.0            # opponent cannot move: no swing
                continue
            with th.no_grad():
                vr = net(th.from_numpy(np.stack([r.encode(-side) for r in replies])).to(device))
            total += -float(th.max(vr))  # opponent plays their best reply
        value = total / len(DICE21)
        if value > best_value:
            best_value, best_board = value, cand
    return best_board


def play_game(net, device, collect: bool = False, rng=None, eps: float = 0.0, search: str = "1ply"):
    rng = rng or np.random.default_rng()
    board = Board()
    side = WHITE if rng.random() < 0.5 else BLACK
    traj: list[tuple[np.ndarray, int]] = []
    plies = 0
    while True:
        w = board.winner()
        if w is not None:
            outcome = 1.0 if w == WHITE else -1.0
            return board, plies, outcome, traj, side
        if plies > 500:
            return board, plies, 0.0, traj, side
        d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        pips = [d1, d2] if d1 != d2 else [d1] * 4
        if collect:
            traj.append((board.encode(side), side))
        if eps and rng.random() < eps:
            cands = board.legal_moves(side, pips)
            board = cands[int(rng.integers(len(cands)))]
        elif search == "2ply":
            board = choose_2ply(board, side, pips, net, device)
        else:
            board = choose(board, side, pips, net, device)
        side = -side
        plies += 1


def train(games: int, lam: float = 0.7, alpha: float = 1e-3, hidden: int = 256, device: str = "cuda"):
    net = ValueNet(hidden).to(device)
    opt = th.optim.Adam(net.parameters(), lr=alpha)
    rng = np.random.default_rng(0)
    t0 = time.time()
    recent = []
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    for g in range(1, games + 1):
        board, plies, outcome, traj, _ = play_game(net, device, collect=True, rng=rng)
        if not traj:
            continue
        xs = np.stack([t[0] for t in traj])
        signs = np.array([t[1] for t in traj], dtype=np.float32)
        targets = outcome * signs                       # outcome from each state's mover view
        x = th.from_numpy(xs).to(device)
        y = th.from_numpy(targets).to(device)
        v = net(x)
        # TD(λ): blend bootstrapped values with the final outcome
        with th.no_grad():
            v_next = th.cat([v.detach()[1:], th.zeros(1, device=device)])
            td_target = (1 - lam) * v_next + lam * y
        loss = ((v - td_target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        th.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        recent.append(abs(outcome))
        if g % 500 == 0:
            th.save({"state_dict": net.state_dict(), "hidden": hidden}, MODEL)
            print(f"game {g:>6,} | plies {plies:3d} | loss {loss.item():.4f} | "
                  f"{g / (time.time() - t0):6.1f} games/s", flush=True)
    th.save({"state_dict": net.state_dict(), "hidden": hidden}, MODEL)
    print("saved", MODEL)


def load(device: str = "cuda"):
    ck = th.load(MODEL, map_location=device, weights_only=False)
    net = ValueNet(ck["hidden"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def evaluate(net, device, n: int = 200) -> dict:
    rng = np.random.default_rng(123)
    wins = 0
    pips = []
    for _ in range(n):
        board, plies, outcome, _, first = play_game(net, device, rng=rng)
        # net started as `first`; outcome is +1 if WHITE won
        if (outcome > 0 and first == WHITE) or (outcome < 0 and first == BLACK):
            wins += 1
        pips.append(plies)
    return {"games": n, "win_rate_first_player": wins / n, "avg_plies": float(np.mean(pips))}


def human_play(net, device, human_side: str) -> None:
    board = Board()
    side = WHITE
    rng = np.random.default_rng()
    while board.winner() is None:
        d1, d2 = rng.integers(1, 7), rng.integers(1, 7)
        dice = (int(d1), int(d2))
        print(f"\ndice {dice} | side {'WHITE (you)' if side == WHITE else 'BLACK (bot)'}")
        print_board(board)
        if (human_side == "white") == (side == WHITE):
            cands = board.legal_moves(side, dice)
            for i in range(len(cands)):
                print(f"  [{i}] next board above")
            idx = int(input(f"choose move 0..{len(cands)-1}: "))
            board = cands[idx]
        else:
            board = choose(board, side, dice, net, device)
        side = -side
    print_board(board)
    print("winner:", "WHITE" if board.winner() == WHITE else "BLACK")


def print_board(b: Board) -> None:
    top = " ".join(f"{i+1:>3}" for i in range(12, 24))
    bot = " ".join(f"{i+1:>3}" for i in range(11, -1, -1))
    print(top); print(" ".join(f"{int(b.points[i]):>3}" for i in range(12, 24)))
    print(" ".join(f"{int(b.points[i]):>3}" for i in range(11, -1, -1))); print(bot)
    print(f"bar W/B {b.bar_w}/{b.bar_b} | off W/B {b.off_w}/{b.off_b}")


def self_check(games: int = 60) -> None:
    net = ValueNet().to("cpu").eval()
    rng = np.random.default_rng(0)
    ends = {WHITE: 0, BLACK: 0, 0: 0}
    for _ in range(games):
        board, plies, outcome, _, _ = play_game(net, "cpu", rng=rng)
        w = board.winner()
        ends[w if w is not None else 0] += 1
        assert board.off_w <= 15 and board.off_b <= 15, (board.off_w, board.off_b)
        assert board.points.max() <= 15 and board.points.min() >= -15
    print(f"{games} games completed: white {ends[WHITE]} black {ends[BLACK]} unfinished {ends[0]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=0)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--eval", type=int, default=0)
    p.add_argument("--play", action="store_true")
    p.add_argument("--human", default="white", choices=["white", "black"])
    p.add_argument("--lam", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if args.check:
        self_check(args.games)
    elif args.train:
        train(args.train, lam=args.lam, alpha=args.alpha, device=args.device)
    elif args.eval:
        net = load(args.device)
        print(evaluate(net, args.device, args.eval))
    elif args.play:
        human_play(load(args.device), args.device, args.human)
    else:
        self_check(args.games)
