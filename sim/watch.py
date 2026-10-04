"""Watch a sim-trained net play in the GPU simulator, live in the browser, at any speed.

    uv run python -m sim.watch --run s1 [--port 8096] [--bots 4] [--arenas 4]

Runs its own small simulator (`--arenas` arenas of 16 players) next to training: the first slots
play the run's latest checkpoint (reloaded whenever training saves a new one), the last
`--bots` slots are the scripted sim bot. The page at http://127.0.0.1:<port>/ draws the
arenas side by side (click one to zoom in) and sets the speed (1x = real time, 80 ms per step;
"max" = as fast as the GPU goes). While training runs, a step costs ~45 ms whatever the number of
arenas (small GPU batches wait behind training's), so max is ~1.7x per arena: more arenas is the
cheap way to see more play.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from rl.common import ROOT
from rl.ppo_sim import from_checkpoint
from sim.bots import scripted_actions
from sim.engine import EJECT_R2, FOOD_R2
from sim.env import SimEnv

PAGE = Path(__file__).with_name("watch.html")
STEP_S = 0.08                      # one env step = 2 server ticks of 40 ms


class Watcher:
    def __init__(self, run: str, bots: int, vram_gb: float, arenas: int, players: int, past: int):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2 ** 30 / total))
        self.dir = ROOT / "runs" / "rl" / run
        self.env = SimEnv(B=arenas, P=players, episode_steps=10 ** 9, curriculum_p=0.0)
        self.bots = bots
        # slots: [latest net | past league snapshots (`past` slots, 3 snapshots) | scripted bots]
        self.past = past
        self.groups, self.names, self.ver, self.ready = [], [], None, False
        self.speed = 1.0           # x real time; 0 = max
        self.paused = False
        self.model, self.ckpt = None, ""
        self.frame = "{}"
        self.lock = threading.Lock()
        self.steps = 0
        self.rate = 0.0
        self.new_round = False     # restart the arena(s): on every new checkpoint and on the page's button
        self.round, self.round_step = 0, 0
        self.last_view = 0.0       # last /frame request: no open page -> the sim idles (training gets the GPU)

    def lineup(self) -> list[tuple[str, Path]]:
        """(name, checkpoint) per model group. An evolution run (evo.json): every current member
        plus the newest hall-of-fame entry. A plain run: its latest net, then past league
        snapshots (the oldest, a middle one and the newest but one)."""
        evo = self.dir / "evo.json"
        if evo.exists():
            members = json.loads(evo.read_text())["members"]
            out = []
            for m in members:
                ck = self.dir / m["id"] / "latest.pt"
                out.append((m["id"], ck if ck.exists() else Path(m["ckpt"])))
            hof = sorted((self.dir / "hof").glob("*.pt"))
            if hof:
                out.append((f"hof {hof[-1].stem}", hof[-1]))
            return out
        ck = self.dir / "latest.pt"
        if not ck.exists():
            return []
        out = [(f"{self.dir.name} now", ck)]
        snaps = sorted((self.dir / "league").glob("it*.pt"))
        if snaps and self.past:
            for f in sorted({snaps[0], snaps[len(snaps) // 2], snaps[max(0, len(snaps) - 2)]}):
                out.append((f"{self.dir.name} it{int(f.stem[2:])}", f))
        return out

    def version(self) -> tuple:
        """What counts as "new models available": for an evolution run a finished generation, for a
        plain run a new checkpoint."""
        evo = self.dir / "evo.json"
        if evo.exists():
            return ("gen", json.loads(evo.read_text())["gen"])
        ck = self.dir / "latest.pt"
        return ("ckpt", ck.stat().st_mtime if ck.exists() else 0)

    def maybe_reload(self, force: bool = False) -> None:
        """Models change only at the start of a round: a newer version just lights up the page's
        "new round" button (self.ready), unless nothing is loaded yet or `force` (the button)."""
        ver = self.version()
        if self.groups and not force:
            self.ready = ver != self.ver
            return
        lu = [(n, p) for n, p in self.lineup() if p.exists()]
        if not lu:
            return
        self.ver, self.ready = ver, False
        P, nb = self.env.P, self.bots
        n_play = P - nb
        evo = (self.dir / "evo.json").exists()
        if evo:                                    # equal shares
            shares = [n_play // len(lu) + (1 if i < n_play % len(lu) else 0) for i in range(len(lu))]
        else:                                      # latest net gets everything but `past` slots
            k = len(lu) - 1
            past = self.past if k else 0
            shares = [n_play - past] + [past // k + (1 if i < past % k else 0) for i in range(k)]
        self.groups, self.names, lo = [], [], 0
        for (name, path), n in zip(lu, shares):
            m = from_checkpoint(str(path), "cuda").eval()
            m.binary_eps = 0.0
            self.groups.append((m, lo, lo + n))
            self.names += [name] * n
            lo += n
        self.names += ["bot"] * nb
        it = torch.load(lu[0][1], map_location="cpu", weights_only=False).get("it", "?")
        self.ckpt = f"{self.dir.name} generation {ver[1]}: {len(lu)} models" if evo else f"{self.dir.name} it {it}"
        self.model = self.groups[0][0]
        self.new_round = True

    def snapshot(self) -> str:
        # One GPU->CPU copy for everything: every separate .cpu() is a sync that waits for a GPU
        # time slice behind training, and dozens of them per frame made the viewer crawl.
        w = self.env.w
        parts = [w.calive, w.cx, w.cy, w.cr2, w.falive, w.fx, w.fy, w.ealive, w.ex, w.ey,
                 w.valive, w.vx, w.vy, w.vr2]
        flat = torch.cat([t.float().reshape(-1) for t in parts]).cpu()
        out, i = [], 0
        for t in parts:
            out.append(flat[i:i + t.numel()].view(t.shape))
            i += t.numel()
        self._host = out
        return json.dumps({
            "map": w.map, "ckpt": self.ckpt, "speed": self.speed, "rate": round(self.rate, 1),
            "step": self.steps, "bots": self.bots, "paused": self.paused, "names": self.names,
            "round": self.round, "ready": self.ready, "round_s": round(self.round_step * STEP_S),
            "food_r": FOOD_R2 ** 0.5, "eject_r": EJECT_R2 ** 0.5,
            "arenas": [self.arena(b) for b in range(self.env.B)],
        }, separators=(",", ":"))

    def arena(self, b: int) -> dict:
        P, K = self.env.P, self.env.w.K
        cal, cx, cy, cr2, fal, fx, fy, eal, ex, ey, val, vx, vy, vr2 = (t[b] for t in self._host)
        al = cal > 0
        r = cr2.sqrt()
        cells = [[round(float(cx[i])), round(float(cy[i])), round(float(r[i]), 1), i // K]
                 for i in al.nonzero()[:, 0].tolist()]
        fa, ea, va = fal > 0, eal > 0, val > 0
        mass = torch.zeros(P)
        mass.index_add_(0, torch.arange(P * K) // K, cr2 / 100 * al)
        xy = lambda x, y, m: torch.stack([x[m], y[m]], 1).round().int().reshape(-1).tolist()
        return {
            "cells": cells, "food": xy(fx, fy, fa), "eject": xy(ex, ey, ea),
            "virus": [[round(float(a)), round(float(b_)), round(float(c), 1)]
                      for a, b_, c in zip(vx[va], vy[va], vr2[va].sqrt())],
            "mass": [round(float(m)) for m in mass],
        }

    def loop(self) -> None:
        g, s, _, _, _ = self.env.reset()
        P, nb = self.env.P, self.bots
        last_reload, t_rate, n_rate, last_snap = 0.0, time.time(), 0, 0.0
        while True:
            now = time.time()
            if now - last_reload > 5:
                self.maybe_reload()
                last_reload = now
            if time.time() - self.last_view > 5:
                time.sleep(0.2)
                continue
            if self.new_round:
                if self.groups and self.ready:     # the button: pick up the newest models first
                    self.maybe_reload(force=True)
                g, s, _, _, _ = self.env.reset()
                self.new_round = False
                self.round += 1
                self.round_step = 0
            if self.paused or self.model is None:
                time.sleep(0.05)
                with self.lock:
                    self.frame = self.snapshot()
                continue
            t0 = time.time()
            with torch.no_grad():
                a = torch.zeros(self.env.B, P, 3, dtype=torch.long, device=g.device)
                gv, sv = g.view(self.env.B, P, *g.shape[1:]), s.view(self.env.B, P, -1)
                for m, lo, hi in self.groups:
                    a[:, lo:hi] = m.act(gv[:, lo:hi].reshape(-1, *g.shape[1:]),
                                        sv[:, lo:hi].reshape(-1, s.shape[-1]), amp=True)[0].view(self.env.B, hi - lo, 3)
                a = a.reshape(-1, 3)
                if nb:
                    a = a.view(self.env.B, P, 3)
                    a[:, P - nb:] = scripted_actions(self.env.w, self.env.bot_split_prob)[:, P - nb:]
                    a = a.reshape(-1, 3)
            g, s, _, _, _ = self.env.step(a)
            self.steps += 1
            self.round_step += 1
            n_rate += 1
            if time.time() - t_rate > 1:
                self.rate, n_rate, t_rate = n_rate * STEP_S / (time.time() - t_rate), 0, time.time()
            if time.time() - last_snap > 1 / 15:   # ~15 frames/s is plenty: don't sync the GPU every step
                last_snap = time.time()
                frame = self.snapshot()
                with self.lock:
                    self.frame = frame
            if self.speed > 0:
                time.sleep(max(0.0, STEP_S / self.speed - (time.time() - t0)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="s1", help="a ppo_sim run, or an rl.evo population (e.g. e1)")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--bots", type=int, default=4, help="slots played by the scripted sim bot")
    p.add_argument("--arenas", type=int, default=1)
    p.add_argument("--players", type=int, default=24, help="per arena (training uses 24)")
    p.add_argument("--past", type=int, default=6, help="slots played by past league snapshots")
    p.add_argument("--vram-gb", type=float, default=2.0)
    args = p.parse_args()
    wt = Watcher(args.run, args.bots, args.vram_gb, args.arenas, args.players, args.past)

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
                f = wt.dir / "progress.json"
                self.send(f.read_bytes() if f.exists() else b"[]", "application/json")
            elif self.path.startswith("/newround"):
                wt.new_round = True
                self.send(b"ok", "text/plain")
            elif self.path.startswith("/turbo"):
                # pause this run's training (rl.ppo_sim waits while PAUSE exists) and run flat out
                flag = wt.dir / "PAUSE"
                if flag.exists():
                    flag.unlink()
                    wt.speed = 1.0
                else:
                    flag.touch()
                    wt.speed = 0.0
                self.send(b"ok", "text/plain")
            elif self.path.startswith("/speed"):
                v = self.path.split("=", 1)[-1]
                if v == "pause":
                    wt.paused = not wt.paused
                else:
                    wt.speed = float(v)
                self.send(b"ok", "text/plain")
            else:
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")

    threading.Thread(target=wt.loop, daemon=True).start()
    print(f"sim watch: http://127.0.0.1:{args.port}/  (run {args.run})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
