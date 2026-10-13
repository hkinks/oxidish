"""Run a match's rule-bot agents with the Rust client (rust/arena_client) instead of the Python one.

    cargo build --release -j 2 --manifest-path rust/arena_client/Cargo.toml
    uv run python -m arena.rust_runner --agents route:4 champ=runs/final_champion.json:4 --ticks 3000 --seed 1

Same server, same lockstep protocol, same row format as arena.tournament (`--out` JSON); also reachable as
`arena.tournament --rust`. Only rule bots (route / field / name=genome.json) are supported.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .match import ROOT, free_port, start_server

_EXE = "arena_client.exe" if os.name == "nt" else "arena_client"


def binary() -> Path:
    path = Path(os.environ.get("ARENA_CLIENT_BIN", ROOT / "rust" / "arena_client" / "target" / "release" / _EXE))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: cargo build --release -j 2 --manifest-path rust/arena_client/Cargo.toml")
    return path


def supported(specs: list[str]) -> bool:
    """True when every lineup entry is a rule bot (the Rust runner has no NN / Jev / hybrid policies)."""
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
    path = Path(os.environ.get("OGAR_BIN", ROOT / "rust" / "ogar" / "target" / "release" / ("ogar.exe" if os.name == "nt" else "ogar")))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: cargo build --release -j 2 --manifest-path rust/ogar/Cargo.toml")
    return path


def play_ogar(specs: list[str], ticks: int, sync: int, seed: int, bots: int = 8, map_size: int = 7000,
              food_mult: float = 1.0) -> list[dict]:
    """One match in the Rust engine (rust/ogar): the whole world + agents in one process, no websockets."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.json"
        cmd = [str(ogar_binary()), "match", "--ticks", str(ticks), "--sync", str(sync), "--seed", str(seed),
               "--bots", str(bots), "--map", str(map_size), "--food-mult", str(food_mult),
               "--out", str(out), "--agents", *specs]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"ogar failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return json.loads(out.read_text(encoding="utf-8"))


def play_rust(specs: list[str], ticks: int, port: int, sync: int, seed: int) -> list[dict]:
    """Run the lineup against a server already listening on `port`; one row per agent."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.json"
        cmd = [str(binary()), "run", "--port", str(port), "--ticks", str(ticks), "--sync", str(sync),
               "--seed", str(seed), "--out", str(out), "--agents", *specs]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"arena_client failed: {proc.stderr.strip() or proc.stdout.strip()}")
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
    from .match import parse_lineup
    n = len(parse_lineup(args.agents))
    port = free_port()
    server = start_server(port, args.bots, args.seed, n, args.sync, args.map, food_mult=args.food_mult)
    try:
        started = time.monotonic()
        rows = play_rust(args.agents, args.ticks, port, args.sync, args.seed)
        wall = time.monotonic() - started
    finally:
        server.terminate()
    for r in rows:
        print(f"{r['policy']:<10} avg {r['avg_mass']:8.1f} peak {r['peak_mass']:7.0f} deaths {r['deaths']} kills {r['kills']}")
    print(f"{args.ticks} ticks in {wall:.1f}s wall = {args.ticks * 0.04 / wall:.1f}x realtime")
    if args.out:
        Path(args.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
