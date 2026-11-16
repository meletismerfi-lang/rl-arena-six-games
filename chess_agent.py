"""Chess bot: Stockfish behind a UCI wrapper with an adjustable strength dial.

  # best move for a position
  .venv/Scripts/python.exe chess_agent.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  # play a game against it, you as white (type moves in SAN or UCI)
  .venv/Scripts/python.exe chess_agent.py --play --human white --elo 1600
  # self-play a game and score it
  .venv/Scripts/python.exe chess_agent.py --selfplay 3

Strength: --elo 0..3000 maps to Stockfish's UCI_LimitStrength/UCI_Elo; --depth/--movetime
override it for full strength (default movetime 1.0s ≈ 3,400+ Elo).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent


def find_binary() -> str:
    cands = glob.glob("C:/Users/melet/AppData/Local/Microsoft/WinGet/**/stockfish*.exe", recursive=True)
    cands += [c for c in glob.glob(str(ROOT / "engines" / "stockfish*"))]
    if not cands:
        raise SystemExit("stockfish binary not found (winget install Stockfish.Stockfish)")
    return cands[0]


class ChessBot:
    def __init__(self, elo: int = 0, depth: int | None = None, movetime: float | None = 1.0,
                 threads: int = 4, hash_mb: int = 512):
        self.engine = chess.engine.SimpleEngine.popen_uci(find_binary())
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.elo, self.depth, self.movetime = elo, depth, movetime
        self._apply_strength()

    def _apply_strength(self) -> None:
        if self.elo and self.elo < 3200:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, self.elo)})
        else:
            self.engine.configure({"UCI_LimitStrength": False})

    def limit(self) -> chess.engine.Limit:
        if self.depth:
            return chess.engine.Limit(depth=self.depth)
        if self.movetime:
            return chess.engine.Limit(time=self.movetime)
        return chess.engine.Limit(depth=20)

    def move(self, board: chess.Board, legal_fallback: bool = True) -> chess.Move:
        result = self.engine.play(board, self.limit())
        if result.move is None:
            raise SystemExit("engine returned no move")
        return result.move

    def evaluate(self, board: chess.Board) -> chess.engine.PovScore:
        return self.engine.analyse(board, chess.engine.Limit(depth=18))["score"]

    def close(self) -> None:
        self.engine.quit()


def describe(board: chess.Board, move: chess.Move, bot: ChessBot) -> str:
    san = board.san(move)
    score = bot.evaluate(board)
    return f"{san:<7} eval {score.white():>10}"


def play_human(bot: ChessBot, human: str, elo: int | None) -> None:
    board = chess.Board()
    human_white = human == "white"
    while not board.is_game_over():
        print("\n" + str(board) + f"\n\nFEN: {board.fen()}\n")
        if board.turn == chess.WHITE if human_white else board.turn == chess.BLACK:
            san = input("your move (SAN or UCI, 'quit' to stop): ").strip()
            if san.lower() in {"quit", "q"}:
                return
            try:
                move = board.parse_san(san)
            except ValueError:
                try:
                    move = chess.Move.from_uci(san)
                except ValueError:
                    print("illegal move")
                    continue
            if move not in board.legal_moves:
                print("illegal move")
                continue
            board.push(move)
        else:
            move = bot.move(board)
            print(f"bot: {describe(board, move, bot)}")
            board.push(move)
    print("result:", board.result(), board.outcome())


def selfplay(bot: ChessBot, games: int) -> None:
    for g in range(1, games + 1):
        board = chess.Board()
        plies = 0
        while not board.is_game_over() and plies < 300:
            board.push(bot.move(board))
            plies += 1
        print(f"game {g}: {board.result():5s} in {plies} plies ({board.outcome().termination.name})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fen", default=None)
    p.add_argument("--play", action="store_true")
    p.add_argument("--human", default="white", choices=["white", "black"])
    p.add_argument("--selfplay", type=int, default=0)
    p.add_argument("--elo", type=int, default=0, help="0 = full strength")
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--movetime", type=float, default=1.0)
    args = p.parse_args()

    bot = ChessBot(elo=args.elo, depth=args.depth, movetime=args.movetime)
    try:
        if args.selfplay:
            selfplay(bot, args.selfplay)
        elif args.play:
            play_human(bot, args.human, args.elo)
        else:
            board = chess.Board(args.fen) if args.fen else chess.Board()
            move = bot.move(board)
            print(f"position: {board.fen()}")
            print(f"best move: {move.uci()} ({board.san(move)})")
            print(f"eval: {bot.evaluate(board).white()}")
    finally:
        bot.close()


if __name__ == "__main__":
    main()
