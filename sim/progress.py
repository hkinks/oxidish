"""Is the run still improving? Every `--every` minutes, the run's newest checkpoint plays a sim
tournament (sim.tourney) against a fixed set of its own past checkpoints, in shared 50-player
arenas. The references never change, so the newest net's scores over time form a learning curve.

    uv run python -m sim.progress --run s1 --refs runs/rl/s1/league/it00100.pt runs/rl/s1/resume_00240.pt

Results: runs/rl/<run>/progress.json (one row per evaluation; sim.watch charts it) and
TensorBoard run `<run>-eval`.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from rl.common import ROOT
from sim.tourney import run as tourney


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="s1")
    p.add_argument("--refs", nargs="+", required=True, help="fixed reference checkpoints")
    p.add_argument("--every", type=float, default=15.0, help="minutes between evaluations")
    p.add_argument("--b", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500, help="80 ms each: 1500 = 2 min of game time")
    p.add_argument("--vram-gb", type=float, default=4.0)
    p.add_argument("--backfill", type=int, default=0, help="first evaluate every Nth saved ckpt_*.pt (0 = none)")
    args = p.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2 ** 30 / total))
    rdir = ROOT / "runs" / "rl" / args.run
    out = rdir / "progress.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    tb = SummaryWriter(str(ROOT / "logs" / "tb" / f"{args.run}-eval"))
    refs = {Path(r).stem: r for r in args.refs}
    done = {r["it"] for r in rows}
    for r in rows:                             # (re)write earlier results so the charts are complete
        for k, v in r.items():
            if isinstance(v, dict):
                tb.add_scalars("models/avg_mass", {k: v["avg_mass"]}, r["it"])
                tb.add_scalars("models/deaths", {k: v["deaths"]}, r["it"])
                tb.add_scalars("models/kills", {k: v["kills"]}, r["it"])
    queue = []
    if args.backfill:
        cks = sorted(rdir.glob("ckpt_*.pt"))[::args.backfill]
        queue = [(int(c.stem[5:]), c) for c in cks if int(c.stem[5:]) not in done]
    while True:
        if queue:
            it, ck = queue.pop(0)
        else:
            ck = rdir / "latest.pt"
            it = torch.load(ck, map_location="cpu", weights_only=False).get("it", 0) if ck.exists() else None
        if it is not None and it not in done:
            snap = rdir / "progress_eval.pt"          # latest.pt may be rewritten mid-evaluation
            snap.write_bytes(ck.read_bytes())
            res = tourney({"latest": str(snap), **refs}, B=args.b, steps=args.steps, seed=1000 + it)
            row = {"it": it, "time": time.time(), **{k: v for k, v in res.items()}}
            rows.append(row)
            rows.sort(key=lambda r: r["it"])
            done.add(it)
            out.write_text(json.dumps(rows, indent=1))
            best_ref = max(v["log_mass"] for k, v in res.items() if k != "latest")
            tb.add_scalar("eval/log_mass", res["latest"]["log_mass"], it)
            tb.add_scalar("eval/margin_vs_best_ref", res["latest"]["log_mass"] - best_ref, it)
            tb.add_scalar("eval/avg_mass", res["latest"]["avg_mass"], it)
            tb.add_scalar("eval/deaths", res["latest"]["deaths"], it)
            tb.add_scalar("eval/kills", res["latest"]["kills"], it)
            for k, v in res.items():          # every model in the same matches, one chart per metric
                tb.add_scalars("models/avg_mass", {k: v["avg_mass"]}, it)
                tb.add_scalars("models/deaths", {k: v["deaths"]}, it)
                tb.add_scalars("models/kills", {k: v["kills"]}, it)
            tb.flush()
            print(f"it {it}: " + "  ".join(f"{k} {v['log_mass']:.2f} ({v['avg_mass']:.0f} mass, "
                                           f"{v['deaths']:.2f} deaths, {v['kills']:.2f} kills)"
                                           for k, v in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"])),
                  flush=True)
        if not queue:
            time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
