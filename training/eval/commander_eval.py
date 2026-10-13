"""Commander on vs off for one base bot (docs/commander.md).

    uv run python -m training.eval.commander_eval --cand plan=bots/specs/plan_c34.json --backend heuristic --worlds 4 --game-min 20 --speed 4
    uv run python -m training.eval.commander_eval --cand plan=bots/specs/plan_c34.json --backend off      --worlds 4 --game-min 20

Each world is an `oxidish-server serve` (own ports, never the live ones) with the candidate, the usual rule-bot field (champ, r2g9,
safe, ramp_r2, ramp300) and 8 server bots. With a backend, a Commander commands the candidate over the control port in
real time at --speed (LLM latency counts in real time, so use --speed 1 for LLM backends). Reports the share of samples
in which the candidate is the heaviest player of the whole world, and its mean mass (dead = 0).
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .commander import Commander, Control, make_backend
from .endless import ROOT, TICKS_PER_MIN, ogar_bin
from .util import free_port, limit_cpus

MAIN = Path(os.environ.get("AGAR_MAIN_REPO", "C:/Users/hkink/Workspace/agario-bot"))
FIELD = ["champ=bots/specs/final_champion.json", "r2g9=bots/specs/r2/champion_009.json", "safe=bots/specs/champ_safe.json",
         "ramp_r2=bots/specs/r2g9_ramp800.json", "ramp300=bots/specs/r2g9_ramp300.json"]


def run_world(cand: str, backend: str, seed: int, game_min: float, speed: float, food_mult: float, log_s: float,
              interval: float, tag: str) -> dict:
    name = cand.split("=")[0]
    port, cport = free_port(), free_port()
    env = dict(os.environ)
    env.setdefault("ORT_DYLIB_PATH", str(MAIN / "runs/vision/ort/onnxruntime.dll"))
    env.setdefault("OXIDISH_DLL_DIRS", os.pathsep.join([str(MAIN / ".venv/Lib/site-packages/torch/lib"), str(MAIN / "runs/vision/ort")]))
    cmd = [str(ogar_bin()), "serve", "--port", str(port), "--bind", "127.0.0.1", "--bots", "8", "--map", "7000",
           "--food-mult", str(food_mult), "--seed", str(seed), "--max-tps", str(25 * speed if backend != "off" else 0),
           "--log-players", str(log_s), "--control-port", str(cport), "--agents", f"{cand}:1", *[f"{f}:1" for f in FIELD]]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", env=env)
    samples, cm, be = [], None, None
    end_tick = int(game_min * TICKS_PER_MIN)
    try:
        if backend != "off":
            time.sleep(3)
            be = make_backend(backend)
            log = ROOT / "logs" / f"commander-eval-{tag}-{seed}.jsonl"
            log.parent.mkdir(exist_ok=True)
            log.unlink(missing_ok=True)
            cm = Commander(Control(port=cport), be, [f"{name}-0"], interval, log)
            cm.run()
        for line in proc.stdout:
            if line.startswith("[players] "):
                rec = json.loads(line[len("[players] "):])
                samples.append(rec)
                if rec["t"] >= end_tick:
                    break
    finally:
        if cm:
            cm.stop.set()
        proc.kill()
        proc.wait()
    return {"seed": seed, "samples": samples, "decisions": cm.decisions if cm else 0, "errors": cm.errors if cm else 0,
            "calls": be.calls if be else 0}


def summarize(worlds: list[dict], name: str, skip_min: float) -> dict:
    lead = n = 0
    masses = []
    per_world = []
    for w in worlds:
        wl = wn = 0
        wm = []
        for rec in w["samples"]:
            if rec["t"] < skip_min * TICKS_PER_MIN or not rec["p"]:
                continue
            top = max(rec["p"], key=lambda p: p["m"])
            wn += 1
            wl += top["n"].split("-")[0] == name
            wm.append(sum(p["m"] for p in rec["p"] if p["n"].split("-")[0] == name))
        lead, n = lead + wl, n + wn
        masses += wm
        per_world.append({"seed": w["seed"], "leader_share": round(100 * wl / max(wn, 1), 1), "mean_mass": round(sum(wm) / max(len(wm), 1))})
    return {"leader_share": round(100 * lead / max(n, 1), 1), "mean_mass": round(sum(masses) / max(len(masses), 1)),
            "samples": n, "per_world": per_world, "decisions": sum(w["decisions"] for w in worlds),
            "llm_calls": sum(w["calls"] for w in worlds), "errors": sum(w["errors"] for w in worlds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="name=spec.json of the base bot")
    ap.add_argument("--backend", choices=["off", "heuristic", "jev", "anthropic"], required=True)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--game-min", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.0, help="x real time when a commander is on (off runs unthrottled)")
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--seed-base", type=int, default=51000)
    ap.add_argument("--skip-min", type=float, default=2.0)
    ap.add_argument("--food-mult", type=float, default=2.0)
    ap.add_argument("--log-s", type=float, default=5.0)
    ap.add_argument("--cpus", type=int, default=11)
    ap.add_argument("--out")
    a = ap.parse_args()
    limit_cpus(a.cpus)
    name = a.cand.split("=")[0]
    t0 = time.time()
    res = [None] * a.worlds

    def go(i):
        res[i] = run_world(a.cand, a.backend, a.seed_base + i, a.game_min, a.speed, a.food_mult, a.log_s, a.interval, f"{name}-{a.backend}")

    ts = [threading.Thread(target=go, args=(i,)) for i in range(a.worlds)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    out = {"cand": a.cand, "backend": a.backend, "worlds": a.worlds, "game_min": a.game_min, "speed": a.speed,
           "wall_s": round(time.time() - t0), **summarize(res, name, a.skip_min)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
