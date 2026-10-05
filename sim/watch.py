"""Watch sim-trained nets play, live in the browser.

    uv run python -m sim.watch --run e1        # an rl.evo population: every member + the newest hall of fame
    uv run python -m sim.watch --run s1        # a plain rl.ppo_sim run: its latest net + 3 past snapshots

Runs its own simulator arena next to training (it is not a replay of training games), at real
time: 80 ms per step. Faster is not available while training runs. Each step is ~100 small GPU
jobs that wait behind training's, ~45 ms whatever the arena size. The last `--bots` slots are the
scripted sim bot. When newer models exist (an evolution generation finished, or a plain run saved
a checkpoint) the page's "new round" button lights up; pressing it loads them and restarts the
arena with everyone at mass 10. With no page open for 5 s the arena stops, so training gets
the GPU.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from rl.common import ROOT
from rl.ppo_sim import from_checkpoint, obs_for, obs_kind
from sim.bots import scripted_actions
from sim.engine import EJECT_R2, FOOD_R2
from sim.env import SimEnv

PAGE = Path(__file__).with_name("watch.html")
STEP_S = 0.08                      # one env step = 2 server ticks of 40 ms


class Watcher:
    def __init__(self, run: str, players: int, bots: int, past: int, vram_gb: float):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2 ** 30 / total))
        self.dirs = [ROOT / "runs" / "rl" / r for r in run.split(",")]   # several runs share the arena
        self.dir = self.dirs[0]
        self.env = SimEnv(B=1, P=players, episode_steps=10 ** 9, curriculum_p=0.0)
        self.bots, self.past = bots, past
        self.groups, self.names, self.ver, self.ready, self.title = [], [], None, False, ""
        self.paused, self.new_round = False, False
        self.held = None                   # the last actions (frame-skip nets hold theirs)
        self.round, self.round_step, self.rate = 0, 0, 0.0
        # per slot this round, kept on the GPU (no sync per step): splits, ejects (W), deaths, and the
        # current life's length in steps (mass/min = mass gained / life time)
        self.counts = torch.zeros(4, players, device="cuda")
        self.last_view = 0.0       # time of the last /frame request
        self.frame = "{}"
        self.lock = threading.Lock()

    @property
    def evo(self) -> bool:
        return (self.dir / "evo.json").exists()

    def lineup(self) -> list[tuple[str, Path]]:
        """(name, checkpoint) per model, over all runs."""
        out = []
        for d in self.dirs:
            pre = f"{d.name}/" if len(self.dirs) > 1 else ""
            if (d / "evo.json").exists():
                for m in json.loads((d / "evo.json").read_text())["members"]:
                    ck = d / m["id"] / "latest.pt"
                    out.append((pre + m["id"], ck if ck.exists() else Path(m["ckpt"])))
                hof = sorted((d / "hof").glob("*.pt"))
                if hof:
                    out.append((f"{pre}hof {hof[-1].stem}", hof[-1]))
                continue
            ck = d / "latest.pt"
            if not ck.exists():
                continue
            out.append((f"{d.name} now", ck))
            snaps = sorted((d / "league").glob("it*.pt"))
            if snaps and self.past and len(self.dirs) == 1:   # the oldest, a middle one and the newest but one
                for f in sorted({snaps[0], snaps[len(snaps) // 2], snaps[max(0, len(snaps) - 2)]}):
                    out.append((f"{d.name} it{int(f.stem[2:])}", f))
        return out

    def version(self) -> tuple:
        """Per run: an evolution's generation, a plain run's latest checkpoint time."""
        v = []
        for d in self.dirs:
            if (d / "evo.json").exists():
                v.append(("gen", json.loads((d / "evo.json").read_text())["gen"]))
            else:
                ck = d / "latest.pt"
                v.append(("ckpt", ck.stat().st_mtime if ck.exists() else 0))
        return tuple(v)

    def check(self) -> None:
        """Note newer models (lights the button); load models if none are loaded yet."""
        if self.groups:
            self.ready = self.version() != self.ver
        else:
            self.load()

    def load(self) -> None:
        lu = [(n, p) for n, p in self.lineup() if p.exists()]
        if not lu:
            return
        self.ver, self.ready = self.version(), False
        n_play = self.env.P - self.bots
        if self.evo:                                   # equal shares
            shares = [n_play // len(lu) + (i < n_play % len(lu)) for i in range(len(lu))]
        else:                                          # the latest net gets all but `past` slots
            k = len(lu) - 1
            past = self.past if k else 0
            shares = [n_play - past] + [past // k + (i < past % k) for i in range(k)]
        self.groups, self.names, lo = [], [], 0
        for (name, path), n in zip(lu, shares):
            m = from_checkpoint(str(path), "cuda").eval()
            m.binary_eps = 0.0
            self.groups.append((m, lo, lo + n))
            self.names += [name] * n
            lo += n
        kinds = {obs_kind(m) for m, _, _ in self.groups}
        want = "both" if len(kinds) > 1 else kinds.pop()
        if want != self.env.obs:       # entity nets need entity observations
            self.env = SimEnv(B=1, P=self.env.P, episode_steps=10 ** 9, curriculum_p=0.0, obs=want)
            self.held = None
        self.names += ["bot"] * self.bots
        if len(self.dirs) > 1:
            self.title = f"{' + '.join(d.name for d in self.dirs)}: {len(lu)} models"
        elif self.evo:
            self.title = f"{self.dir.name} generation {self.ver[0][1]}: {len(lu)} models"
        else:
            it = torch.load(lu[0][1], map_location="cpu", weights_only=False).get("it", "?")
            self.title = f"{self.dir.name} it {it}"
        self.new_round = True

    def snapshot(self) -> str:
        # One GPU->CPU copy for everything: each separate .cpu() is a sync that waits for a GPU
        # time slice behind training.
        w, K = self.env.w, self.env.w.K
        parts = [w.calive, w.cx, w.cy, w.cr2, w.falive, w.fx, w.fy, w.ealive, w.ex, w.ey,
                 w.valive, w.vx, w.vy, w.vr2, self.counts[None]]
        flat = torch.cat([t[0].float().reshape(-1) for t in parts]).cpu()
        host, i = [], 0
        for t in parts:
            host.append(flat[i:i + t[0].numel()])
            i += t[0].numel()
        cal, cx, cy, cr2, fal, fx, fy, eal, ex, ey, val, vx, vy, vr2, counts = host
        counts = counts.view(4, -1)
        al, fa, ea, va = cal > 0, fal > 0, eal > 0, val > 0
        r = cr2.sqrt()
        mass = torch.zeros(self.env.P).index_add_(0, torch.arange(len(cr2)) // K, cr2 / 100 * al)
        xy = lambda x, y, m: torch.stack([x[m], y[m]], 1).round().int().reshape(-1).tolist()
        return json.dumps({
            "map": w.map, "title": self.title, "rate": round(self.rate, 1), "paused": self.paused,
            "names": self.names, "bots": self.bots, "round": self.round, "ready": self.ready,
            "round_s": round(self.round_step * STEP_S), "food_r": FOOD_R2 ** 0.5, "eject_r": EJECT_R2 ** 0.5,
            "cells": [[round(float(cx[j])), round(float(cy[j])), round(float(r[j]), 1), j // K]
                      for j in al.nonzero()[:, 0].tolist()],
            "food": xy(fx, fy, fa), "eject": xy(ex, ey, ea),
            "virus": [[round(float(a)), round(float(b)), round(float(c), 1)]
                      for a, b, c in zip(vx[va], vy[va], vr2[va].sqrt())],
            "mass": [round(float(m)) for m in mass],
            "splits": counts[0].int().tolist(), "ejects": counts[1].int().tolist(),
            "deaths": counts[2].int().tolist(),
            "life_s": [round(float(x) * STEP_S, 1) for x in counts[3]],
        }, separators=(",", ":"))

    def actions(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        P = self.env.P
        if self.held is None:
            self.held = torch.zeros(P, 3, dtype=torch.long, device=g.device)
        a = self.held
        for m, lo, hi in self.groups:
            if self.round_step % m.skip == 0:     # frame-skip nets decide every m.skip steps
                a[lo:hi] = m.act(obs_for(m, self.env, g)[lo:hi], s[lo:hi], amp=True)[0]
            else:                                 # ... holding the move; split / eject fire once
                a[lo:hi, 1:] = 0
        if self.bots:
            a[P - self.bots:] = self.env.bot_actions()[0, P - self.bots:]
        return a

    def count(self, a: torch.Tensor, died: torch.Tensor) -> None:
        alive = self.env.prev_alive.reshape(-1)
        self.counts[0] += ((a[:, 1] > 0) & alive).float()
        self.counts[1] += ((a[:, 2] > 0) & alive).float()
        self.counts[2] += died.float()
        alive_now = self.env.prev_alive.reshape(-1)
        self.counts[3] = torch.where(alive_now, self.counts[3] + 1, torch.zeros_like(self.counts[3]))

    def loop(self) -> None:
        g, s, _, _, _ = self.env.reset()
        last_check, t_rate, n_rate, last_snap = 0.0, time.time(), 0, 0.0
        while True:
            if time.time() - last_check > 5:
                self.check()
                last_check = time.time()
            if time.time() - self.last_view > 5 or not self.groups:   # nobody watching
                time.sleep(0.2)
                continue
            if self.new_round:
                if self.ready:
                    self.load()
                g, s, _, _, _ = self.env.reset()
                self.new_round = False
                self.counts.zero_()
                self.round += 1
                self.round_step = 0
            if self.paused:
                time.sleep(0.05)
                frame = self.snapshot()
            else:
                t0 = time.time()
                with torch.no_grad():
                    a = self.actions(g, s)
                    g, s, _, d, _ = self.env.step(a)
                    self.count(a, d)
                self.round_step += 1
                n_rate += 1
                if time.time() - t_rate > 1:
                    self.rate, n_rate, t_rate = n_rate * STEP_S / (time.time() - t_rate), 0, time.time()
                frame = None
                if time.time() - last_snap > 1 / 15:     # ~15 frames/s: don't sync the GPU every step
                    last_snap = time.time()
                    frame = self.snapshot()
                time.sleep(max(0.0, STEP_S - (time.time() - t0)))
            if frame:
                with self.lock:
                    self.frame = frame

    def progress(self) -> list[dict]:
        """Chart rows {x, label, series: {name: avg mass}}: per generation for an evolution run
        (its best member, the median member, the hall-of-fame yardstick), per evaluated
        checkpoint for a plain run (sim.progress: the newest net vs fixed references)."""
        if self.evo:
            rows = []
            for e in json.loads((self.dir / "evo.json").read_text())["log"]:
                res = e["results"]
                mem = sorted(v["avg_mass"] for k, v in res.items() if k != "hof_last")
                series = {"best": mem[-1], "median": mem[len(mem) // 2]}
                if "hof_last" in res:
                    series["hall of fame"] = res["hof_last"]["avg_mass"]
                rows.append({"x": e["gen"], "label": f"generation {e['gen']}", "series": series})
            return rows
        f = self.dir / "progress.json"
        if not f.exists():
            return []
        return [{"x": r["it"], "label": f"it {r['it']}",
                 "series": {k: v["avg_mass"] for k, v in r.items() if isinstance(v, dict)}}
                for r in json.loads(f.read_text())]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="e1", help="an rl.evo / rl.pbt population (e.g. p1) or a ppo_sim run (e.g. s1); "
                   "several comma-separated (e.g. fp1,fe1) share the arena")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--players", type=int, default=50, help="training arenas have 50")
    p.add_argument("--bots", type=int, default=6, help="slots played by the scripted sim bot")
    p.add_argument("--past", type=int, default=12, help="plain runs: slots played by past snapshots")
    p.add_argument("--vram-gb", type=float, default=2.0)
    args = p.parse_args()
    wt = Watcher(args.run, args.players, args.bots, args.past, args.vram_gb)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/frame"):
                wt.last_view = time.time()
                with wt.lock:
                    body = wt.frame.encode()
                self.send(body, "application/json")
            elif self.path.startswith("/progress"):
                self.send(json.dumps(wt.progress()).encode(), "application/json")
            elif self.path.startswith("/newround"):
                wt.new_round = True
                self.send(b"ok", "text/plain")
            elif self.path.startswith("/pause"):
                wt.paused = not wt.paused
                self.send(b"ok", "text/plain")
            else:
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")

    threading.Thread(target=wt.loop, daemon=True).start()
    print(f"sim watch: http://127.0.0.1:{args.port}/  (run {args.run})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
