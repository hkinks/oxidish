"""Keep the newest checkpoint of every training run playing on the showcase server.

    uv run python -m arena.autojoin --port 9500 --runs g10 g11 --interval 1200

Every --interval seconds, for each run (runs/rl/<run>/) with a checkpoint newer than
the one on the server, the checkpoint is copied to runs/rl/<run>/show_latest_<it>.pt and
joined as "<run>-<it>" (2 copies); the run's previous agents then leave. Run names
can also be added later: any new dir listed in runs/rl/autojoin.txt (one run per line)
is picked up.

Agents are detached processes (PIDs in runs/rl/autojoin.json), so they survive
restarts of this script and of the training runs; only a newer checkpoint of the
same run replaces them.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RL = ROOT / "runs" / "rl"
STATE = RL / "autojoin.json"
EXTRA = RL / "autojoin.txt"
DETACH = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
          if sys.platform == "win32" else 0)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def alive(pid: int) -> bool:
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the process on Windows
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def newest(run: str) -> Path | None:
    ckpts = sorted((RL / run).glob("ckpt_[0-9]*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--interval", type=int, default=1200)
    p.add_argument("--copies", type=int, default=2)
    args = p.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    while True:
        runs = list(args.runs)
        if EXTRA.exists():
            runs += [r.strip() for r in EXTRA.read_text().splitlines() if r.strip()]
        for run in dict.fromkeys(runs):
            ck = newest(run)
            if ck is None:
                continue
            it = int(ck.stem.split("_")[1])
            cur = state.get(run)
            if cur and cur["it"] == it and alive(cur["pid"]):
                continue
            show = RL / run / f"show_latest_{it:05d}.pt"
            try:
                shutil.copy(ck, show)  # training may delete/overwrite; play from a stable copy
            except OSError as e:
                log(f"{run}: copy failed ({e}); retry next round")
                continue
            name = f"{run}-{it}"
            out = (ROOT / "logs" / f"join-auto-{run}.txt").open("w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "arena.join", "--port", str(args.port),
                 "--agents", f"{name}={show}:{args.copies}"],
                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=DETACH)
            if cur:
                stop(cur["pid"])
                old = Path(cur["file"])
                if old != show and old.exists():
                    old.unlink(missing_ok=True)
            state[run] = {"pid": proc.pid, "it": it, "file": str(show), "name": name}
            STATE.write_text(json.dumps(state, indent=1))
            log(f"joined {name} ({args.copies}x){' replacing ' + cur['name'] if cur else ''}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
