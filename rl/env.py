"""Multi-agent arena environment for RL, one per worker process.

Each env runs its own lockstep arena server with `n_learn` learner slots (actions
come from the trainer), heuristic opponents (evolved champions) and the server's
built-in bots. One env step = one lockstep step (`sync` ticks).

Objective: time-integrated log(mass / 10), the same thing evolution optimises
(bigger and longer-lived both pay). Paying log mass every step makes returns very
noisy, so the reward is its telescoped equivalent: each step pays the *change* in
log mass times REWARD_SCALE (= scale / (1 - gamma)), and death pays minus the
log mass held (that income stops). Same objective, reward only when something
happens. Dead steps are invalid (masked out of training) until the slot respawns.

Teacher mode (behaviour cloning): learner slots are driven by a heuristic
teacher; its actions are returned as labels.
"""

import asyncio
import math
import random

import numpy as np

from arena.client import StepConn
from arena.match import free_port, start_server
from arena.policies import build

from . import actions as A
from .guided import C as GC, G as GG, S as GS, GuidedEncoder
from .obs import C as PC, G as PG, S as PS, encode

CURRICULUM_P = 0.5            # share of learner respawns that start big
CURRICULUM_MASS = (50, 5000)  # start mass range for those (log-uniform)
# Reward = telescoped time-integrated mass, in "log" (original) or "linear" mode.
# Linear values being big in proportion (10k is worth 100x 100), which is what a human
# aims for; log made 10 -> 100 worth as much as 1k -> 10k and favoured playing safe small.
REWARD_MODE = "linear"
REWARD_SCALE = {"log": 5.0,       # 0.01 / (1 - 0.998)
                "linear": 1 / 200}
SPAWN_MASS = 10.0
EXPLORE_CELL = 500.0     # novelty grid: first visit to each 500x500 square in a life pays
STUCK_WINDOW, STUCK_DIST = 25, 150.0  # moved < 150 units in 2 s = stuck (logged, see arena/stuck.py)


class ArenaEnv:
    def __init__(self, idx: int, n_learn: int, opponents: list[dict], n_opp: int,
                 episode_ticks: int, bots: int, map_size: int, sync: int,
                 teacher: dict | None = None, seed: int = 0, guide: dict | None = None,
                 opp_every: int = 2, league: str | None = None, league_p: float = 0.5,
                 virus_reward: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 explore_reward: float = 0.0):
        self.idx, self.n_learn, self.n_opp = idx, n_learn, n_opp
        self.pool = opponents
        self.episode_ticks, self.bots, self.map, self.sync = episode_ticks, bots, map_size, sync
        self.teacher_spec = teacher
        self.guide_spec = guide  # guided (residual) nets: per-slot champion + motion obs
        self.opp_every = opp_every  # opponents re-decide every n-th step (1 = full strength)
        self.league, self.league_p = league, league_p  # dir of frozen snapshots of the learner
        # Bonus per virus we shot, per enemy that exploded on our shot, and per unit of
        # its mass (in mass-reward units): shooting pays on its own, hitting pays more.
        self.virus_reward = virus_reward
        self.explore_reward = explore_reward  # per newly visited square this life
        self.rng = random.Random(seed * 1000 + idx)
        self.server = None
        self.resets = 0
        self.learn, self.opp = [], []
        self.n_steps = 0

    async def _open(self):
        self.resets += 1
        port = free_port()
        seed = self.rng.randrange(1, 2 ** 31)
        n_agents = self.n_learn + self.n_opp
        self.server = start_server(port, self.bots, seed, n_agents, self.sync, self.map)
        url = f"ws://127.0.0.1:{port}"
        self.learn = [StepConn(url, f"nn{self.idx}-{i}", self.sync) for i in range(self.n_learn)]
        specs = [self.rng.choice(self.pool) for _ in range(self.n_opp)]
        if self.league:  # league play: part of the opponents are past versions of the learner
            from pathlib import Path
            snaps = sorted(Path(self.league).glob("*.pt"))[-6:]
            specs = [{"kind": "nn", "ckpt": str(self.rng.choice(snaps))}
                     if snaps and self.rng.random() < self.league_p else sp for sp in specs]
        self.opp = [(StepConn(url, f"opp{self.idx}-{i}", self.sync), build(sp))
                    for i, sp in enumerate(specs)]
        self.teachers = [build(self.teacher_spec) for _ in range(self.n_learn)] if self.teacher_spec else None
        self.encoders = ([GuidedEncoder(self.guide_spec) for _ in range(self.n_learn)]
                         if self.guide_spec else None)
        for c in self.learn + [c for c, _ in self.opp]:
            await c.open()
        self.start_tick = None
        self.prev_alive = [False] * self.n_learn
        self.prev_logm = [0.0] * self.n_learn  # reward level (log or linear mass) at the last step
        self.life = [self._new_life() for _ in range(self.n_learn)]
        self.prev_virus = [(0, 0, 0.0)] * self.n_learn
        self.frames = [None] * self.n_learn  # (cx, cy, ext) of the last observation
        self.teacher_acts = np.full((self.n_learn, 3), -1, np.int64)  # teacher's pending action

    async def _close(self):
        for c in self.learn + [c for c, _ in self.opp]:
            try:
                await c.close()
            except Exception:
                pass
        if self.server:
            self.server.terminate()
            self.server = None

    async def _advance(self):
        await asyncio.gather(*(c.next_step() for c in self.learn + [c for c, _ in self.opp]))
        if self.start_tick is None:
            self.start_tick = self.learn[0].tick

    def _observe(self):
        n = self.n_learn
        C, S, G = (GC, GS, GG) if self.encoders else (PC, PS, PG)
        grids = np.zeros((n, C, G, G), np.float32)
        scals = np.zeros((n, S), np.float32)
        rew = np.zeros(n, np.float32)
        done = np.zeros(n, bool)
        valid = np.zeros(n, bool)
        labels = np.full((n, 3), -1, np.int64)
        finished = []
        self.snaps = []
        for i, c in enumerate(self.learn):
            s = c.snapshot()
            self.snaps.append(s)
            alive = not s.dead and s.main is not None
            if alive:
                g, sc, cx, cy, ext = self.encoders[i].encode(s) if self.encoders else encode(s)
                grids[i], scals[i] = g, sc
                self.frames[i] = (cx, cy, ext)
                mass = s.total_mass
                logm = math.log(max(mass, 1.0) / SPAWN_MASS)
                level = logm if REWARD_MODE == "log" else mass  # quantity whose time-integral we maximise
                if self.prev_alive[i]:
                    rew[i] = (level - self.prev_logm[i]) * REWARD_SCALE[REWARD_MODE]
                self.prev_logm[i] = level
                valid[i] = True
                L = self.life[i]
                L["ret"] += logm * 0.001  # logged metric, same scale as earlier runs
                L["peak"] = max(L["peak"], mass)
                L["steps"] += 1
                me = s.main
                cell = (int(me.x // EXPLORE_CELL), int(me.y // EXPLORE_CELL))
                if cell not in L["seen"]:
                    L["seen"].add(cell)
                    if L["steps"] > 1:  # the spawn square is free
                        rew[i] += self.explore_reward
                trail = L["trail"]
                trail.append((me.x, me.y))
                if len(trail) > STUCK_WINDOW:
                    trail.pop(0)
                    if math.hypot(me.x - trail[0][0], me.y - trail[0][1]) < STUCK_DIST:
                        L["stuck"] += 1
                if self.teachers:
                    x, y, sp, ej, *_ = (*self.teachers[i].act(s), False)
                    labels[i] = A.encode(x, y, sp, ej, cx, cy)
            elif self.prev_alive[i]:
                done[i] = True
                rew[i] = -self.prev_logm[i] * REWARD_SCALE[REWARD_MODE]  # that income stops
                finished.append(dict(self.life[i], died=True))
                self.life[i] = self._new_life()
                if self.teachers:
                    self.teachers[i].reset()
                if self.encoders:
                    self.encoders[i].reset()
            shots, pops, pop_mass = c.virus
            ps, pp, pm = self.prev_virus[i]
            self.prev_virus[i] = c.virus
            if (alive or self.prev_alive[i]) and (shots > ps or pops > pp):
                ws, wp, wm = self.virus_reward
                rew[i] += ws * (shots - ps) + wp * (pops - pp) + wm * (pop_mass - pm) / 200
                L = self.life[i] if alive else finished[-1]
                L["shots"] += shots - ps
                L["pops"] += pops - pp
            self.prev_alive[i] = alive
        self.teacher_acts = labels
        return grids, scals, rew, done, valid, labels, finished

    @staticmethod
    def _new_life() -> dict:
        return {"ret": 0.0, "peak": 0.0, "steps": 0, "shots": 0, "pops": 0, "stuck": 0,
                "seen": set(), "trail": []}

    async def reset(self):
        await self._close()
        await self._open()
        for c in self.learn + [c for c, _ in self.opp]:
            await c.act(0, 0)  # first ack spawns everyone
        await self._advance()
        return self._observe()

    async def step(self, acts: np.ndarray):
        sends = []
        self.n_steps += 1
        for i, c in enumerate(self.learn):
            if not self.prev_alive[i] and self.encoders is not None and c.snapshot().dead:
                # Curriculum: often respawn big, so fights/splits/viruses get practised.
                if self.rng.random() < CURRICULUM_P:
                    lo, hi = CURRICULUM_MASS
                    mass = math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
                    await c.set_spawn_size(math.sqrt(100 * mass))
                else:
                    await c.set_spawn_size(0)
            if self.teachers and self.prev_alive[i]:
                move, sp, ej = (int(v) for v in self.teacher_acts[i])
                x, y, split, eject = A.decode(move, sp, ej, *self.frames[i])
            elif self.prev_alive[i] and self.frames[i]:
                dec = self.encoders[i].decode if self.encoders else A.decode
                x, y, split, eject = dec(int(acts[i, 0]), int(acts[i, 1]), int(acts[i, 2]),
                                              *self.frames[i])
            else:
                x, y, split, eject = 0, 0, False, False
            sends.append(c.act(x, y, split, eject))
        for c, pol in self.opp:
            s = c.snapshot()
            if s.dead or s.main is None:
                sends.append(c.act(0, 0))
            elif self.n_steps % self.opp_every:  # cheaper opponents that re-decide less often
                sends.append(c.ack())
            else:
                x, y, split, *rest = pol.act(s)
                sends.append(c.act(x, y, split, bool(rest and rest[0])))
        await asyncio.gather(*sends)
        await self._advance()
        out = list(self._observe())
        if self.learn[0].tick - self.start_tick >= self.episode_ticks:
            # Episode over: close the arena; lives in progress count as finished (truncated).
            for i in range(self.n_learn):
                if self.prev_alive[i]:
                    out[6].append(dict(self.life[i], died=False))
                    out[3][i] = True
            nxt = await self.reset()
            out[0], out[1], out[4], out[5] = nxt[0], nxt[1], nxt[4], nxt[5]
        return tuple(out)


def worker(remote, cfg: dict) -> None:
    """Subprocess loop: ('reset'|'step', data) -> observation tuple."""
    loop = asyncio.new_event_loop()
    env = ArenaEnv(**cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(loop.run_until_complete(env.reset()))
            elif cmd == "step":
                remote.send(loop.run_until_complete(env.step(data)))
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        loop.run_until_complete(env._close())
        remote.close()
