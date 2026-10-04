"""Watch training output; add promising new bots to the live showcase server.

    uv run python -m rl.promote --run p1 [--port 9500] [--interval 600]

Every `--interval` seconds:
- Neural nets: the newest checkpoint of runs/rl/<run>/ plays `--matches` 4v4 matches
  against the latest evolved champion on fresh seeds. If its mean log-mass
  advantage beats every earlier checkpoint's (and --min-adv), it is saved as best.pt and joins
  the showcase as nn<iteration> (replacing the previous nn entry; nnBC stays).
- Evolution: any new runs/r*/champion_###.json joins as e<run>g<gen>.
Decisions are appended to logs/promote.txt.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from torch.utils.tensorboard import SummaryWriter

from arena.tournament import _play

from .common import ROOT, latest_champion


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (ROOT / "logs" / "promote.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def join(port: int, name: str, path: str, count: int = 2) -> subprocess.Popen:
    out = (ROOT / "logs" / f"join-{name}.txt").open("w")
    return subprocess.Popen([sys.executable, "-m", "arena.join", "--port", str(port), "--agents",
                             f"{name}={path}:{count}"], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def evaluate(pool, ckpt: str, champ_path: str, matches: int, ticks: int, seed_base: int):
    specs = [f"cand={ckpt}:4", f"champ={champ_path}:4"]
    futures = [pool.submit(_play, i, specs, ticks, 8, 7000, 2, seed_base) for i in range(matches)]
    rows = [r for f in futures for r in f.result()]
    diffs, wins = [], 0
    for seed in {r["seed"] for r in rows}:
        c = statistics.fmean(r["log_mass"] for r in rows if r["seed"] == seed and r["policy"] == "cand")
        h = statistics.fmean(r["log_mass"] for r in rows if r["seed"] == seed and r["policy"] == "champ")
        diffs.append(c - h)
        wins += c > h
    return statistics.fmean(diffs), wins / len(diffs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="p1")
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--matches", type=int, default=6)
    p.add_argument("--ticks", type=int, default=4500)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--min-adv", type=float, default=-0.25,
                   help="never promote nets worse than this log-mass gap to the champion")
    args = p.parse_args()

    run_dir = ROOT / "runs" / "rl" / args.run
    seen_champs = set(str(p) for p in ROOT.glob("runs/[rq]*/champion_[0-9]*.json"))
    evaluated: set[str] = set()
    best = float("-inf")
    nn_proc = None
    procs = []
    pool = ProcessPoolExecutor(args.workers)
    round_no = 0
    tb = SummaryWriter(str(ROOT / "logs" / "tb" / args.run))  # same run dir as training
    try:
        while True:
            for path in sorted(ROOT.glob("runs/[rq]*/champion_[0-9]*.json")):
                if str(path) not in seen_champs:
                    seen_champs.add(str(path))
                    name = f"e{path.parent.name}g{int(path.stem.split('_')[1])}"
                    procs.append(join(args.port, name, str(path)))
                    log(f"new evolved champion {path.name} ({path.parent.name}) joined as {name}")
            ckpts = sorted(run_dir.glob("ckpt_*.pt"))
            if ckpts and str(ckpts[-1]) not in evaluated:
                ck = ckpts[-1]
                evaluated.add(str(ck))
                champ = latest_champion()
                champ_path = ROOT / "runs" / "rl" / "eval_champion.json"
                champ_path.write_text(json.dumps(champ))
                round_no += 1
                # Evaluate a private copy: training may rotate or delete its checkpoints.
                cand = ROOT / "runs" / "rl" / f"eval_{args.run}_{ck.name}"
                try:
                    shutil.copy(ck, cand)
                    adv, win = evaluate(pool, str(cand), str(champ_path), args.matches, args.ticks,
                                        700_000 + 1000 * round_no)
                except Exception as exc:  # keep watching; a failed round is just skipped
                    log(f"{ck.name}: evaluation failed ({exc!r}), skipped")
                    time.sleep(60)
                    continue
                promising = adv > max(best, args.min_adv)
                it_no = int(ck.stem.split("_")[1])
                tb.add_scalar("eval/log_mass_vs_champion", adv, it_no)
                tb.add_scalar("eval/win_rate_vs_champion", win, it_no)
                tb.flush()
                log(f"{ck.name}: log-mass vs champion {adv:+.3f}, win rate {win:.0%}"
                    f"{'  -> PROMOTED to showcase' if promising else ''}")
                if promising:
                    best = adv
                    shutil.copy(cand, run_dir / "best.pt")
                    keep = run_dir / f"show_{ck.stem}.pt"  # stable file for the joined agents
                    shutil.copy(cand, keep)
                    if nn_proc:
                        nn_proc.terminate()
                    it = int(ck.stem.split("_")[1])
                    nn_proc = join(args.port, f"nn{it}", str(keep))
                cand.unlink(missing_ok=True)
            time.sleep(args.interval if ckpts else 60)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
