# Credits and provenance

## Who wrote this

**Agent:** Hermes Agent — an autonomous coding agent built by [Nous Research](https://nousresearch.com).
**Model:** `deepseek-v4.1-flash:cloud` (runtime reported by the agent's own environment on 16 November 2026).

The agent wrote the code, ran the training jobs, measured the results, and reported them. It was
executed as a tool-using process on the operator's own hardware, driving a shell, a GPU, and a
local model-training loop.

## Human review and supervision

**Every step of this work was performed under human review and supervision.**

Concretely, the human operator:

- set every goal and every constraint ("30M parameters", "60M steps", "use ViZDoom", "make a game
  to play against it", "make also AI vs AI", "upload our work");
- approved each change of direction, including killing runs, switching environments, and
  spending money on a rented GPU;
- supplied assets (the chess piece set) and credentials (GitHub, RunPod) themselves;
- monitored live: GPU utilisation, VRAM, power draw, running state, and window output;
- inspected and drove the running applications (arena, chess, backgammon, Doom) and reported
  defects — several fixes in this repository exist only because the operator watched the result
  and said it was wrong (the arena re-launching a game from a stale click; Doom's action space).

The agent did not act unsupervised on anything with an external side effect: every push, every
paid compute run, and every published artefact required the operator's explicit instruction.

## Third-party assets

| asset | source | licence |
|---|---|---|
| Chess piece art (`pieces_svg/`, rasterised to `pieces_png/`) | Uray M. János, "Chess Pieces" SVG set | CC BY-SA (attribution kept in `pieces_svg/*.svg` headers) |
| Chess engine | [Stockfish](https://stockfishchess.org/) | GPL-3.0 |
| ENABLE word list (`data/enable1.txt`, 172,823 words) | public-domain word list maintained by Alan Beale | public domain |
| Atari ROMs | bundled with `ale-py` 0.10.1 (Arcade Learning Environment) | see ALE / ROM licences |
| Doom scenarios | ViZDoom scenario `.cfg` files | ViZDoom is MIT; Doom engine is GPL-2.0 |

Referenced stock imagery the operator supplied (a watermarked iStock backgammon sheet) was used
only as a **visual reference**; the backgammon board, checkers and dice in `backgammon_ui.py` are
drawn procedurally from scratch, so no stock file is redistributed here.

## Honest scope

- The chess agent is **Stockfish**, not a neural network trained here. Calling it "an RL model"
  would be false; it is an engine wrapped behind a common interface.
- Scrabble is a trie move generator plus rack-leave heuristics. There is no opponent modelling
  and no learning — it is the weakest of the six agents.
- Numbers in the README are from runs recorded on this machine. Where a run failed or regressed,
  that is stated rather than quietly dropped (Pac-Man 20M/30M/10M runs, the Doom 8-action run).
