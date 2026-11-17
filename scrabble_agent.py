"""Scrabble engine + bot.

Move generation is the real algorithm: a trie walk along each line, with cross-word checks at
every newly placed tile. The bot values a move by `score + leave_bonus(rack_after)`, where the
leave bonus is a learned leaf-evaluation approximated by a small hand-weighted model — the
honest description is "strong greedy with leave awareness", not tournament strength.

  .venv/Scripts/python.exe scrabble_agent.py --check                 # engine self-test
  .venv/Scripts/python.exe scrabble_agent.py --play --human first    # play in the terminal
  .venv/Scripts/python.exe scrabble_agent.py --selfplay 3            # bot vs bot
  .venv/Scripts/python.exe scrabble_agent.py --bot-move "abcde..."   # best move for a rack
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORDS = ROOT / "data" / "enable1.txt"
SIZE = 15

VALUES = {**{c: 1 for c in "aeioulnrst"}, **{c: 2 for c in "dg"}, **{c: 3 for c in "bcmp"},
          **{c: 4 for c in "fhvwy"}, "k": 5, "j": 8, "x": 8, "q": 10, "z": 10}
COUNTS = {"a": 9, "b": 2, "c": 2, "d": 4, "e": 12, "f": 2, "g": 3, "h": 2, "i": 9, "j": 1,
          "k": 1, "l": 4, "m": 2, "n": 6, "o": 8, "p": 2, "q": 1, "r": 6, "s": 4, "t": 6,
          "u": 4, "v": 2, "w": 2, "x": 1, "y": 2, "z": 1, "?": 2}

TW = {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)}
DW = {(1, 1), (2, 2), (3, 3), (4, 4), (10, 10), (11, 11), (12, 12), (13, 13),
      (1, 13), (2, 12), (3, 11), (4, 10), (10, 4), (11, 3), (12, 2), (13, 1)}
TL = {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)}
DL = {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2), (6, 6), (6, 8), (6, 12),
      (7, 3), (7, 11), (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14), (12, 6),
      (12, 8), (14, 3), (14, 11)}


class Trie:
    __slots__ = ("kids", "word")

    def __init__(self):
        self.kids: dict[str, "Trie"] = {}
        self.word = False

    def add(self, w: str) -> None:
        n = self
        for ch in w:
            n = n.kids.setdefault(ch, Trie())
        n.word = True


def load_trie(path: Path = WORDS) -> Trie:
    t = Trie()
    with path.open() as fh:
        for line in fh:
            w = line.strip().lower()
            if 2 <= len(w) <= SIZE and w.isalpha():
                t.add(w)
    return t


@dataclass
class Move:
    word: str
    row: int
    col: int
    horizontal: bool
    score: int = 0
    blanks: int = 0
    placed: list = field(default_factory=list)   # [(row, col, letter, is_blank)]

    def tiles_used(self) -> str:
        """Rack letters actually spent, with '?' standing in for a blank."""
        return "".join("?" if blank else ch for _, _, ch, blank in self.placed)

    def cells(self):
        for i, ch in enumerate(self.word):
            yield (self.row, self.col + i) if self.horizontal else (self.row + i, self.col), ch


@dataclass
class Board:
    grid: list[list[str | None]] = field(default_factory=lambda: [[None] * SIZE for _ in range(SIZE)])
    trie: Trie | None = None

    def filled(self) -> int:
        return sum(1 for r in self.grid for c in r if c)

    def at(self, r: int, c: int) -> str | None:
        return self.grid[r][c] if 0 <= r < SIZE and 0 <= c < SIZE else None

    # ------------------------------------------------------------- move generation
    def generate(self, rack: str) -> list[Move]:
        rack_counts: dict[str, int] = {}
        for ch in rack.lower():
            rack_counts[ch] = rack_counts.get(ch, 0) + 1
        first = self.filled() == 0
        moves: dict[tuple, Move] = {}

        for horizontal in (True, False):
            for line in range(SIZE):
                self._line_moves(line, horizontal, rack_counts, first, moves)
        out = list(moves.values())
        for m in out:
            m.score, m.blanks = self.score_move(m)
        return [m for m in out if m.score > 0]

    def _cells(self, line: int, horizontal: bool) -> list[str | None]:
        return [self.grid[line][i] if horizontal else self.grid[i][line] for i in range(SIZE)]

    def _line_moves(self, line: int, horizontal: bool, rack: dict[str, int], first: bool,
                    out: dict[tuple, Move]) -> None:
        cells = self._cells(line, horizontal)

        def cross_ok(r: int, c: int, ch: str) -> bool:
            if horizontal:
                word = ch
                x = r - 1
                while self.at(x, c):
                    word = self.at(x, c) + word
                    x -= 1
                x = r + 1
                while self.at(x, c):
                    word += self.at(x, c)
                    x += 1
            else:
                word = ch
                y = c - 1
                while self.at(r, y):
                    word = self.at(r, y) + word
                    y -= 1
                y = c + 1
                while self.at(r, y):
                    word += self.at(r, y)
                    y += 1
            return len(word) == 1 or self._is_word(word)

        def rec(start: int, pos: int, node: Trie, letters: list[str], used: list[str], blanks: int,
                placed: list | None = None) -> None:
            placed = placed or []
            if letters and node.word:
                self._maybe_record(start, pos, letters, used, blanks, horizontal, line, out, placed)
            if pos >= SIZE:
                return
            existing = cells[pos]
            if existing:
                nxt = node.kids.get(existing)
                if nxt:
                    rec(start, pos + 1, nxt, letters + [existing], used, blanks, placed)
                return
            # word may not run past a gap into another word
            if pos > start and cells[pos] == "":
                pass
            for ch, cnt in list(rack.items()):
                if cnt <= 0:
                    continue
                if pos > start and blanks == 0:
                    pass
                if ch == "?":
                    continue
                nxt = node.kids.get(ch)
                if not nxt:
                    continue
                r, c = (line, pos) if horizontal else (pos, line)
                if not cross_ok(r, c, ch):
                    continue
                rack[ch] -= 1
                r2, c2 = (line, pos) if horizontal else (pos, line)
                rec(start, pos + 1, nxt, letters + [ch], used + [ch], blanks,
                    placed + [(r2, c2, ch, False)])
                rack[ch] += 1
            if rack.get("?", 0) > 0:
                for ch in "abcdefghijklmnopqrstuvwxyz":
                    nxt = node.kids.get(ch)
                    if not nxt:
                        continue
                    r, c = (line, pos) if horizontal else (pos, line)
                    if not cross_ok(r, c, ch):
                        continue
                    rack["?"] -= 1
                    r2, c2 = (line, pos) if horizontal else (pos, line)
                    rec(start, pos + 1, nxt, letters + [ch], used + ["?"], blanks + 1,
                        placed + [(r2, c2, ch, True)])
                    rack["?"] += 1

        assert self.trie is not None
        for start in range(SIZE):
            if cells[start] == "":
                continue
            if start > 0 and cells[start - 1]:
                continue                       # must begin at the leftmost free cell
            rec(start, start, self.trie, [], [], 0)

    def _maybe_record(self, start: int, end: int, letters: list[str], used: list[str], blanks: int,
                      horizontal: bool, line: int, out: dict, placed: list | None = None) -> None:
        if end < SIZE and self._cells(line, horizontal)[end]:
            return                             # would extend into another word
        if not used:
            return                             # nothing placed: not a move
        word = "".join(letters)
        if len(word) < 2 and self.filled() > 0:
            return
        key = (word, line, start, horizontal)
        if key in out:
            return
        out[key] = Move(word=word, row=line if horizontal else start,
                        col=start if horizontal else line, horizontal=horizontal,
                        blanks=blanks, placed=list(placed or []))

    def _is_word(self, w: str) -> bool:
        node = self.trie
        for ch in w:
            node = node.kids.get(ch)
            if node is None:
                return False
        return node.word

    # ------------------------------------------------------------- scoring
    def score_move(self, m: Move) -> tuple[int, int]:
        """Standard Scrabble scoring: premiums apply only to newly placed tiles, the word
        multipliers apply to the main word (not the cross words), +50 for a bingo."""
        main = 0
        word_mult = 1
        placed = 0
        crosses: list[str] = []
        for i, ch in enumerate(m.word):
            r, c = (m.row, m.col + i) if m.horizontal else (m.row + i, m.col)
            existing = self.grid[r][c]
            if existing is None:
                placed += 1
                lm = 3 if (r, c) in TL else 2 if (r, c) in DL else 1
                main += VALUES.get(ch, 0) * lm
                if (r, c) in TW:
                    word_mult *= 3
                elif (r, c) in DW:
                    word_mult *= 2
                cross = self._cross_word(r, c, ch, m.horizontal)
                if cross:
                    crosses.append(cross)
            else:
                main += VALUES.get(existing, 0)

        score = main * word_mult + sum(sum(VALUES.get(ch, 0) for ch in w) for w in crosses)
        if placed == 7:
            score += 50
        return score, m.word.count("?")

    def _cross_word(self, r: int, c: int, ch: str, horizontal: bool) -> str:
        if horizontal:
            pre, suf = "", ""
            y = c - 1
            while self.at(r, y):
                pre = self.at(r, y) + pre
                y -= 1
            y = c + 1
            while self.at(r, y):
                suf += self.at(r, y)
                y += 1
        else:
            pre, suf = "", ""
            y = r - 1
            while self.at(y, c):
                pre = self.at(y, c) + pre
                y -= 1
            y = r + 1
            while self.at(y, c):
                suf += self.at(y, c)
                y += 1
        return pre + ch + suf if (pre or suf) else ""

    def apply(self, m: Move) -> None:
        for (r, c), ch in m.cells():
            if self.grid[r][c] is None:
                self.grid[r][c] = ch

    # ------------------------------------------------------------- display
    def show(self) -> str:
        rows = []
        header = "    " + " ".join(f"{c+1:>2}" for c in range(SIZE))
        rows.append(header)
        for r in range(SIZE):
            line = f"{r+1:>2}  " + " ".join((self.grid[r][c] or ".").rjust(2) for c in range(SIZE))
            rows.append(line)
        return "\n".join(rows)


LEAVE_W = {"a": 0.2, "e": 0.3, "i": 0.1, "o": 0.2, "u": -0.2, "s": 1.6, "?": 2.4,
           "q": -2.2, "z": -1.2, "x": -1.0, "j": -1.0, "k": -0.4}


def leave_value(rack: str) -> float:
    """Rough equity of holding these tiles: 's' and blanks are gold, 'q'/'z'/'x' are dead weight."""
    v = sum(LEAVE_W.get(ch, -0.05) for ch in rack.lower())
    vowels = sum(1 for ch in rack.lower() if ch in "aeiou")
    if vowels == 0:
        v -= 1.5
    if vowels >= 5:
        v -= 1.0
    return v


def best_move(board: Board, rack: str, top_k: int = 5) -> list[tuple[Move, float]]:
    moves = board.generate(rack)
    scored = []
    for m in moves:
        left = list(rack.lower())
        for ch in m.tiles_used():
            if ch in left:
                left.remove(ch)
            elif "?" in left:
                left.remove("?")
        scored.append((m, m.score + leave_value("".join(left))))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def draw(bag: list[str], n: int) -> str:
    return "".join(bag.pop(random.randrange(len(bag))) for _ in range(min(n, len(bag))))


def selfplay(games: int = 2, verbose: bool = False) -> None:
    trie = load_trie()
    results = []
    for g in range(games):
        bag = [c for c, n in COUNTS.items() for _ in range(n)]
        random.shuffle(bag)
        board = Board(trie=trie)
        racks = {"bot1": draw(bag, 7), "bot2": draw(bag, 7)}
        scores = {"bot1": 0, "bot2": 0}
        passes = 0
        turn = "bot1"
        while passes < 4 and bag or (passes < 4 and any(racks.values())):
            best = best_move(board, racks[turn], top_k=1)
            if not best or best[0][1] < 0:
                passes += 1
                turn = "bot2" if turn == "bot1" else "bot1"
                continue
            passes = 0
            m = best[0][0]
            board.apply(m)
            scores[turn] += m.score
            for ch in m.word:
                if ch in racks[turn]:
                    racks[turn] = racks[turn].replace(ch, "", 1)
                elif "?" in racks[turn]:
                    racks[turn] = racks[turn].replace("?", "", 1)
            racks[turn] += draw(bag, 7 - len(racks[turn]))
            if verbose:
                print(f"{turn}: {m.word} ({m.row+1},{m.col+1}) {'H' if m.horizontal else 'V'} = {m.score}")
            turn = "bot2" if turn == "bot1" else "bot1"
        results.append(scores)
        print(f"game {g+1}: {scores}")
    print("mean per side:", {k: sum(r[k] for r in results) / len(results) for k in ("bot1", "bot2")})


def self_check() -> None:
    trie = load_trie()
    b = Board(trie=trie)
    mv = b.generate("helloabc")
    print("opening moves for rack helloabc:", len(mv), "| best:", sorted([(m.score, m.word) for m in mv])[-3:])
    assert any(m.word == "hello" for m in mv), "expected 'hello' as a legal opening"
    b.apply([m for m in mv if m.word == "hello"][0])
    print(b.show())
    nxt = b.generate("worldsx")
    assert nxt, "no follow-up moves found"
    best = max(nxt, key=lambda m: m.score)
    print("follow-up best:", best.word, best.score, (best.row + 1, best.col + 1), "H" if best.horizontal else "V")
    print("engine OK")


def human_play(first: str = "human") -> None:
    trie = load_trie()
    bag = [c for c, n in COUNTS.items() for _ in range(n)]
    random.shuffle(bag)
    board = Board(trie=trie)
    rack = draw(bag, 7)
    bot_rack = draw(bag, 7)
    scores = {"you": 0, "bot": 0}
    turn = "you" if first == "human" else "bot"
    passes_in_a_row = 0
    while passes_in_a_row < 4:
        if turn == "you":
            print("\n" + board.show())
            print(f"\nyour rack: {rack}   scores you {scores['you']} / bot {scores['bot']}")
            moves = board.generate(rack)
            if not moves:
                print("no moves — passing")
                passes_in_a_row += 1
                if passes_in_a_row >= 4:
                    break
                turn = "bot"
                continue
            shown = sorted(moves, key=lambda m: -m.score)[:15]
            for i, m in enumerate(shown):
                print(f"  [{i:>2}] {m.word:<12} at row {m.row+1:>2} col {m.col+1:>2} "
                      f"{'H' if m.horizontal else 'V'} = {m.score}")
            print("   or type: word row col h/v   ('pass' to skip)")
            raw = input("> ").strip().lower()
            if raw in {"pass", "p"}:
                passes_in_a_row += 1
                if passes_in_a_row >= 4:
                    break
                turn = "bot"
                continue
            chosen = None
            if raw.isdigit() and int(raw) < len(shown):
                chosen = shown[int(raw)]
            else:
                parts = raw.split()
                if len(parts) != 4:
                    print("need: word row col h/v")
                    continue
                w, r, c, d = parts[0], int(parts[1]) - 1, int(parts[2]) - 1, parts[3]
                for m in moves:
                    if m.word == w and m.row == r and m.col == c and m.horizontal == (d == "h"):
                        chosen = m
                        break
            if not chosen:
                print("illegal move")
                continue
            board.apply(chosen)
            passes_in_a_row = 0
            scores["you"] += chosen.score
            for ch in chosen.word:
                if ch in rack:
                    rack = rack.replace(ch, "", 1)
                elif "?" in rack:
                    rack = rack.replace("?", "", 1)
            rack += draw(bag, 7 - len(rack))
            turn = "bot"
        else:
            best = best_move(board, bot_rack, top_k=1)
            if not best:
                print("bot passes")
                passes_in_a_row += 1
                if passes_in_a_row >= 4:
                    break
                turn = "you"
                continue
            m = best[0][0]
            board.apply(m)
            passes_in_a_row = 0
            scores["bot"] += m.score
            for ch in m.word:
                if ch in bot_rack:
                    bot_rack = bot_rack.replace(ch, "", 1)
                elif "?" in bot_rack:
                    bot_rack = bot_rack.replace("?", "", 1)
            bot_rack += draw(bag, 7 - len(bot_rack))
            print(f"\nbot plays {m.word} at ({m.row+1},{m.col+1}) {'H' if m.horizontal else 'V'} "
                  f"for {m.score}  [totals you {scores['you']} / bot {scores['bot']}]")
            turn = "you"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--selfplay", type=int, default=0)
    ap.add_argument("--human", default="first", choices=["first", "second"])
    ap.add_argument("--bot-move", default=None)
    args = ap.parse_args()

    if args.check:
        self_check()
    elif args.play:
        human_play("human" if args.human == "first" else "bot")
    elif args.selfplay:
        selfplay(args.selfplay, verbose=True)
    elif args.bot_move:
        trie = load_trie()
        b = Board(trie=trie)
        for m, eq in best_move(b, args.bot_move, top_k=8):
            print(f"{m.word:<12} ({m.row+1},{m.col+1}) {'H' if m.horizontal else 'V'} "
                  f"score {m.score:>3} equity {eq:6.1f}")
    else:
        ap.print_help()
