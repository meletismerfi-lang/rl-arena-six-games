"""Build a self-contained HTML page you can open to watch a trained agent.

  python make_demo.py --game mspacman --algo ppo
  python make_demo.py --game pong --algo dqn
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def svg_curve(steps, means, w=760, h=260, pad=34):
    if len(steps) == 0:
        return "<p>no eval data</p>"
    lo, hi = float(min(means)), float(max(means))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    smin, smax = float(min(steps)), float(max(steps)) or 1.0

    def x(s):
        return pad + (s - smin) / max(smax - smin, 1) * (w - 2 * pad)

    def y(v):
        return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)

    pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in zip(steps, means))
    grid = "".join(
        f'<line x1="{pad}" y1="{y(lo + (hi - lo) * i / 4):.1f}" x2="{w - pad}" y2="{y(lo + (hi - lo) * i / 4):.1f}" '
        f'stroke="#1f2a37" stroke-width="1"/>'
        f'<text x="6" y="{y(lo + (hi - lo) * i / 4) + 4:.1f}" fill="#6b7280" font-size="11">'
        f'{lo + (hi - lo) * i / 4:.0f}</text>'
        for i in range(5)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">'
        f'{grid}<polyline points="{pts}" fill="none" stroke="#22d3ee" stroke-width="2.5"/>'
        f'<text x="{pad}" y="{h - 8}" fill="#6b7280" font-size="11">{int(smin):,} steps</text>'
        f'<text x="{w - pad - 60}" y="{h - 8}" fill="#6b7280" font-size="11">{int(smax):,} steps</text>'
        f'</svg>'
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="mspacman")
    p.add_argument("--algo", default="ppo")
    p.add_argument("--run", default=None, help="run dir name (defaults to algo)")
    p.add_argument("--videos", default=None, help="video dir (defaults to videos/<game>)")
    p.add_argument("--min-frames", type=int, default=100, help="skip stub recordings")
    args = p.parse_args()

    run = args.run or args.algo
    log_dir = ROOT / "logs" / args.game / run
    video_dir = (Path(args.videos) if args.videos else ROOT / "videos" / args.game).resolve()
    eval_npz = log_dir / "eval" / "evaluations.npz"

    def n_frames(p: Path) -> int:
        import cv2
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    videos = []
    if video_dir.exists():
        for v in sorted(video_dir.glob("*.mp4")):
            if n_frames(v) >= args.min_frames:
                videos.append(v)
    if not videos:
        raise SystemExit(f"no videos in {video_dir} — run play.py first")

    steps, means = [], []
    if eval_npz.exists():
        d = np.load(eval_npz)
        steps = d["timesteps"].tolist()
        means = d["results"].mean(axis=1).tolist()

    meta = {}
    for m in (ROOT / "models" / args.game / run).glob("train_meta.json"):
        meta = json.loads(m.read_text())
    steps_txt = f"{meta['steps']:,}" if isinstance(meta.get("steps"), int) else "n/a"
    secs_txt = str(meta.get("seconds", "n/a"))

    # real (unclipped) scores recorded by play.py, if present
    real_scores = []
    sf = video_dir / "scores.json"
    if sf.exists():
        for rec in json.loads(sf.read_text()).values():
            real_scores += [int(s) for s in rec.get("scores", [])]
    score_card = (f'<span class="stat">recorded scores <b>{", ".join(map(str, real_scores))}</b></span>'
                  if real_scores else "")

    best_eval = f"{max(means):.0f}" if means else "n/a"
    cards = ""
    for v in videos[:6]:
        size = v.stat().st_size / 1024
        cards += (
            f'<div class="card"><video controls loop muted playsinline src="{v.relative_to(ROOT).as_posix()}"></video>'
            f'<div class="cap">{v.name} · {size:.0f} KB</div></div>'
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{args.game} · {args.algo.upper()} agent</title>
<style>
 body{{background:#0b0f14;color:#e5e7eb;font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto;
      margin:0;padding:28px;max-width:1000px}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#94a3b8;margin-bottom:22px}}
 h2{{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin:26px 0 10px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
 .card{{background:#111827;border:1px solid #1f2a37;border-radius:10px;overflow:hidden}}
 video{{width:100%;display:block;image-rendering:pixelated;background:#000}}
 .cap{{padding:8px 10px;color:#94a3b8;font-size:12px}}
 .stat{{display:inline-block;background:#111827;border:1px solid #1f2a37;border-radius:8px;
        padding:8px 12px;margin:0 8px 8px 0}}
 .stat b{{color:#22d3ee}}
 code{{background:#111827;padding:2px 6px;border-radius:5px;color:#a5f3fc}}
</style></head><body>
<h1>{args.game} — {args.algo.upper()} agent</h1>
<div class="sub">Trained locally on the RTX 5060 · {steps_txt} env steps ·
 {secs_txt} s · {meta.get('env_id', 'ALE/MsPacman-v5')}</div>
<div>
 <span class="stat">obs <b>4×84×84</b> grayscale frames</span>
 <span class="stat">algo <b>{args.algo.upper()}</b></span>
 <span class="stat">eval (clipped units) <b>{best_eval}</b></span>
 {score_card}
</div>
<h2>Recorded playthroughs</h2>
<div class="grid">{cards}</div>
<h2>Learning curve (eval reward vs env steps)</h2>
{svg_curve(steps, means)}
<h2>Watch it live</h2>
<p>Open a real-time window:</p>
<pre><code>cd C:/Users/melet/AI/rl-games
.venv/Scripts/python.exe play.py --model models/{args.game}/{run}/best_model.zip --live</code></pre>
</body></html>"""

    out = ROOT / f"{args.game}_demo.html"
    out.write_text(html, encoding="utf-8")
    print("demo:", out)


if __name__ == "__main__":
    main()
