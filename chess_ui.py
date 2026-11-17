"""Play chess against the bot in a window.

  .venv/Scripts/python.exe chess_ui.py            # you are White, bot ~3400 Elo
  .venv/Scripts/python.exe chess_ui.py --elo 1200 --human black

Click a piece, then click its destination. Legal moves are highlighted.
Keys: U undo · N new game · +/- bot strength · S save shot · ESC quit
Pieces are letters (K Q R B N P), white on cream, black on charcoal — no font assets needed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import pygame

from chess_agent import ChessBot

ROOT = Path(__file__).resolve().parent
SQ = 78
BOARD_PX = SQ * 8
HUD = 86
LIGHT = (238, 228, 200)
DARK = (120, 150, 110)
SEL = (246, 216, 92)
MOVE_DOT = (90, 90, 130)
LAST = (205, 210, 140)
WHITE_PIECE = (250, 250, 245)
BLACK_PIECE = (38, 40, 46)
PIECE_LETTER = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
                chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}
PIECE_FILE = {chess.KING: "king", chess.QUEEN: "queen", chess.ROOK: "rook",
              chess.BISHOP: "bishop", chess.KNIGHT: "knight", chess.PAWN: "pawn"}
PIECES_PNG = ROOT / "pieces_png"
_piece_cache: dict = {}


def load_piece_images(size: int) -> dict:
    """SVG set rasterised to pieces_png/*.png (see README: cairosvg in WSL).

    Falls back to an empty dict, in which case the board draws letters instead.
    """
    key = size
    if key in _piece_cache:
        return _piece_cache[key]
    imgs = {}
    if PIECES_PNG.exists():
        candidates = sorted(int(f.stem.rsplit("-", 1)[1]) for f in PIECES_PNG.glob("king-w-*.png"))
        pick = min(candidates, key=lambda px: abs(px - size)) if candidates else None
        for pt, name in PIECE_FILE.items():
            for color, suffix in ((chess.WHITE, "w"), (chess.BLACK, "b")):
                f = PIECES_PNG / f"{name}-{suffix}-{pick}.png"
                if f.exists():
                    img = pygame.image.load(str(f)).convert_alpha()
                    imgs[(pt, color)] = pygame.transform.smoothscale(img, (size, size))
    _piece_cache[key] = imgs
    return imgs


def square_px(sq: chess.Square) -> tuple[int, int]:
    file, rank = chess.square_file(sq), chess.square_rank(sq)
    return file * SQ, (7 - rank) * SQ


def square_at(x: int, y: int) -> chess.Square | None:
    if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
        return None
    return chess.square(x // SQ, 7 - y // SQ)


def draw_board(screen, board: chess.Board, font_piece, font_hud, selected, last_move, status, elo,
               piece_imgs=None):
    if piece_imgs is None:
        piece_imgs = load_piece_images(SQ)
    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            x, y = square_px(sq)
            color = LIGHT if (file + rank) % 2 else DARK
            if last_move and sq in (last_move.from_square, last_move.to_square):
                color = LAST
            pygame.draw.rect(screen, color, (x, y, SQ, SQ))
            piece = board.piece_at(sq)
            if piece and piece_imgs and (piece.piece_type, piece.color) in piece_imgs:
                img = piece_imgs[(piece.piece_type, piece.color)]
                screen.blit(img, (x, y))
            elif piece:
                fg = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                shadow = BLACK_PIECE if piece.color == chess.WHITE else WHITE_PIECE
                glyph = PIECE_LETTER[piece.piece_type]
                t = font_piece.render(glyph, True, shadow)
                screen.blit(t, (x + SQ / 2 - t.get_width() / 2 + 2, y + SQ / 2 - t.get_height() / 2 + 2))
                t = font_piece.render(glyph, True, fg)
                screen.blit(t, (x + SQ / 2 - t.get_width() / 2, y + SQ / 2 - t.get_height() / 2))
    if selected is not None:
        x, y = square_px(selected)
        pygame.draw.rect(screen, SEL, (x + 2, y + 2, SQ - 4, SQ - 4), 4)
        for dest in board.legal_moves:
            if dest.from_square == selected:
                dx, dy = square_px(dest.to_square)
                pygame.draw.circle(screen, MOVE_DOT, (dx + SQ // 2, dy + SQ // 2), 9)

    screen.blit(font_hud.render(status, True, (235, 235, 240)), (10, BOARD_PX + 8))
    screen.blit(font_hud.render(f"bot Elo {elo if elo else 'max'}  ·  U undo · N new · +/- strength · ESC quit",
                                True, (140, 150, 165)), (10, BOARD_PX + 34))
    check = " — CHECK" if board.is_check() else ""
    if board.is_game_over():
        check = f" — GAME OVER: {board.result()}"
    screen.blit(font_hud.render(f"FEN {board.fen()[:46]}{check}", True, (110, 120, 135)), (10, BOARD_PX + 58))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="white", choices=["white", "black"])
    ap.add_argument("--elo", type=int, default=0)
    ap.add_argument("--movetime", type=float, default=0.8)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((BOARD_PX, BOARD_PX + HUD))
    pygame.display.set_caption("Chess — you vs the bot")
    font_piece = pygame.font.SysFont("segoeuisymbol,arial", int(SQ * 0.62), bold=True)
    font_hud = pygame.font.SysFont("consolas", 14)
    piece_imgs = load_piece_images(SQ)
    print(f"piece art: {len(piece_imgs)} images" if piece_imgs else "piece art: letters (no pieces_png)")

    bot = ChessBot(elo=args.elo, movetime=args.movetime)
    human_color = chess.WHITE if args.human == "white" else chess.BLACK
    board = chess.Board()
    selected = None
    last_move = None
    status = "your move" if board.turn == human_color else "bot thinking..."
    history: list[chess.Move] = []
    running = True
    pending_bot = board.turn != human_color

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_n:
                    board = chess.Board(); history.clear(); selected = None; last_move = None
                    pending_bot = board.turn != human_color
                    status = "your move" if board.turn == human_color else "bot thinking..."
                elif ev.key == pygame.K_u and history:
                    board.pop()
                    if history:
                        board.pop()
                    history.clear(); selected = None; last_move = None; pending_bot = False
                    status = "your move"
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_MINUS, pygame.K_UNDERSCORE):
                    step = 200 if ev.key in (pygame.K_PLUS, pygame.K_EQUALS) else -200
                    args.elo = max(600, min(3200, (args.elo or 3200) + step))
                    bot.close(); bot = ChessBot(elo=args.elo, movetime=args.movetime)
                elif ev.key == pygame.K_s:
                    out = ROOT / "videos" / "chess_ui.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(screen, str(out))
                    print("saved", out)
            elif ev.type == pygame.MOUSEBUTTONDOWN and board.turn == human_color and not board.is_game_over():
                sq = square_at(*ev.pos)
                if sq is None:
                    continue
                if selected is None:
                    piece = board.piece_at(sq)
                    if piece and piece.color == human_color:
                        selected = sq
                else:
                    mv = chess.Move(selected, sq)
                    if mv.promotion is None:
                        legal = [m for m in board.legal_moves if m.from_square == selected and m.to_square == sq]
                        if legal and any(m.promotion for m in legal):
                            mv = chess.Move(selected, sq, promotion=chess.QUEEN)
                    if mv in board.legal_moves:
                        board.push(mv); history.append(mv)
                        last_move = mv; selected = None; pending_bot = True
                        status = "bot thinking..."
                    else:
                        piece = board.piece_at(sq)
                        selected = sq if (piece and piece.color == human_color) else None

        if pending_bot and not board.is_game_over():
            time.sleep(args.movetime + 0.05)
            mv = bot.move(board)
            board.push(mv); history.append(mv); last_move = mv
            pending_bot = False
            status = "your move"
        elif board.is_game_over():
            status = f"game over: {board.result()}"

        draw_board(screen, board, font_piece, font_hud, selected, last_move, status, args.elo, piece_imgs)
        pygame.display.flip()
        pygame.time.wait(16)

    bot.close()
    pygame.quit()


if __name__ == "__main__":
    main()
