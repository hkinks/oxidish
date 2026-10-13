"""Seeded rule-bot matches in the Rust engine (`oxidish-server match`): the whole world and its agents in one process.

    cargo build --release -p oxidish-server
    uv run python -m training.eval.rust_runner --agents route:4 champ=bots/specs/final_champion.json:4 --ticks 3000 --seed 1

Used by training.eval.tournament (one row per agent, `--out` JSON). Only rule bots (route / field / name=genome.json).
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .util import ROOT


def supported(specs: list[str]) -> bool:
    """True when every lineup entry is a rule bot (no NN checkpoints or Jev policies)."""
    for spec in specs:
        head, _, tail = spec.rpartition(":")
        key = head if tail.isdigit() else spec
        name, _, path = key.partition("=")
        if path:
            if path.endswith(".pt") or path.startswith(("jev:", "greedy:", "guard", "hybrid", "nosplit:")):
                return False
        elif name not in ("route", "field"):
            return False
    return True


def ogar_binary() -> Path:
    path = Path(os.environ.get("OXIDISH_BIN", ROOT / "target" / "release" / ("oxidish-server.exe" if os.name == "nt" else "oxidish-server")))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: cargo build --release -p oxidish-server")
    return path


def play_ogar(specs: list[str], ticks: int, sync: int, seed: int, bots: int = 8, map_size: int = 7000,
              food_mult: float = 1.0) -> list[dict]:
    """One match in the Rust engine (server): the whole world + agents in one process, no websockets."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.json"
        cmd = [str(ogar_binary()), "match", "--ticks", str(ticks), "--sync", str(sync), "--seed", str(seed),
               "--bots", str(bots), "--map", str(map_size), "--food-mult", str(food_mult),
               "--out", str(out), "--agents", *specs]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"oxidish-server failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["route:4", "field:4"])
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bots", type=int, default=8)
    p.add_argument("--map", type=int, default=7000)
    p.add_argument("--sync", type=int, default=2)
    p.add_argument("--food-mult", type=float, default=1.0)
    p.add_argument("--out")
    args = p.parse_args()
    started = time.monotonic()
    rows = play_ogar(args.agents, args.ticks, args.sync, args.seed, args.bots, args.map, args.food_mult)
    wall = time.monotonic() - started
    for r in rows:
        print(f"{r['policy']:<10} avg {r['avg_mass']:8.1f} peak {r['peak_mass']:7.0f} deaths {r['deaths']} kills {r['kills']}")
    print(f"{args.ticks} ticks in {wall:.1f}s wall = {args.ticks * 0.04 / wall:.1f}x realtime")
    if args.out:
        Path(args.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
