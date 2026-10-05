"""Vectorised RL env on the GPU simulator: same actions / reward / respawn as rl/env.py.

    env = SimEnv(B=64, P=16)
    grids, scalars, reward, done, valid = env.reset()            # [B*NL,13,48,48], [B*NL,16], [B*NL] x3
    grids, scalars, reward, done, valid = env.step(actions)      # actions [B*NL, 3] (move, split, eject)

Agent order is arena-major: row = arena * NL + learner. Slots >= n_learn (if any) are driven by
the scripted bot in sim.bots. Move heads 0..15 are headings, 16 = stay, 17 (FOLLOW) = stay.

Reward (all per learner slot, same formulas as rl/env.py):
  mass change / 200 (telescoped), death pays -mass / 200,
  + explore_reward per newly visited 500x500 map square in a life (the spawn square is free),
  + ws * dshots + wp * dpops + wm * dpopmass / 200 for virus_reward = (ws, wp, wm): a "shot" is a virus
    we fed until it fired, a "pop" an enemy cell that exploded on a virus we shot, popmass its mass.
Per-life stats (peak mass, length, shots, pops, stuck steps) accumulate on the GPU for the tracked
slots; `pop_stats()` returns and clears them.

Reward settings are per arena (`set_reward`), so several differently-rewarded runs can share one
env (rl.pbt). `step(actions, observe=False)` advances the game without encoding observations:
frame skip, where the policy decides every `skip` steps (the encoder then scales velocities and
merge timers by `skip`, so the inputs keep their meaning).
"""

import math

import numpy as np
import torch

from .bots import scripted_actions
from .engine import MIN_SPLIT_R2, START_R2, World
from .obs import Encoder

DIRS = 16
CURRICULUM_P = 0.5                 # rl/env.py: share of learner respawns that start big
CURRICULUM_MASS = (50, 5000)       # ... start mass, log-uniform
REWARD_SCALE = 1 / 200             # REWARD_MODE "linear"
EXPLORE_CELL = 500.0
STUCK_WINDOW, STUCK_DIST = 25, 150.0   # moved < 150 units over the last 24 steps
SPAWN_MASS = 10.0
# running sums: lives, died, peak, steps, shots, pops, stuck share, ret, lives with peak >= 10k
N_SUMS = 10


class SimEnv:
    def __init__(self, B: int = 64, P: int = 16, n_learn: int | None = None, device: str = "cuda",
                 seed: int = 0, map_size: float = 7000.0, food_mult: float = 2.0,
                 episode_steps: int = 1500, curriculum_p: float = CURRICULUM_P,
                 curriculum_mass: tuple = CURRICULUM_MASS, max_eject: int = 256,
                 bot_split_prob: float = 0.5, obs_dtype: torch.dtype = torch.float32,
                 virus_reward: tuple = (0.0, 0.0, 0.0), explore_reward: float = 0.0,
                 kill_reward: tuple = (0.0, 0.0), growth: dict | None = None,
                 action_bonus: tuple = (0.0, 0.0), skip: int = 1, max_player_mass: float = 100_000.0):
        self.w = World(B, P, device, map_size, food_mult, max_eject, seed)
        self.B, self.P = B, P
        self.NL = P if n_learn is None else n_learn
        self.dev = self.w.dev
        self.enc = Encoder(self.w, list(range(self.NL)), call_steps=skip)
        self.episode_steps = episode_steps   # may be tens of thousands for very long lives
        # an arena where one player passed this restarts: respawned mass (curriculum) piles up in
        # one player until its 16 max-size cells cover the map and eat every respawn on its first tick
        self.max_player_mass = max_player_mass
        self.curriculum_p, self.curriculum_mass = curriculum_p, curriculum_mass
        self.bot_split_prob = bot_split_prob
        self.obs_dtype = obs_dtype   # float16 halves the (large) grid tensor
        d = self.w.dev
        # per-arena reward weights, [.., B, 1] (set_reward)
        self.r_virus = torch.zeros(3, B, 1, device=d)
        self.r_kill = torch.zeros(2, B, 1, device=d)
        self.r_explore = torch.zeros(B, 1, device=d)
        self.r_bonus = torch.zeros(2, B, 1, device=d)
        self.r_growth = torch.zeros(B, 1, dtype=torch.bool, device=d)
        self.r_par = torch.zeros(3, B, 1, device=d)       # par step, par mass, death cost
        self.set_reward(slice(None), virus_reward, explore_reward, kill_reward, growth, action_bonus)
        self.n_agents = B * self.NL
        self.py_steps = 0
        self.gen = self.w.gen
        d, NL = self.dev, self.NL
        self.ng = int(math.ceil(map_size / EXPLORE_CELL))
        self.track = torch.ones(B, NL, dtype=torch.bool, device=d)    # slots whose lives are logged
        self._tot = torch.zeros(N_SUMS, B, device=d)     # per arena, so runs sharing the env log apart
        self._pmax = torch.zeros(B, device=d)
        self.l_peak, self.l_steps, self.l_shots, self.l_pops, self.l_stuck, self.l_ret = \
            (torch.zeros(B, NL, device=d) for _ in range(6))
        self.seen = torch.zeros(B, NL, self.ng * self.ng, dtype=torch.bool, device=d)
        self._trail = torch.zeros(STUCK_WINDOW, B, NL, 2, device=d)
        self.prev_vstat = torch.zeros(3, B, NL, device=d)
        self.prev_kstat = torch.zeros(2, B, NL, device=d)
        # inactive slots never (re)spawn: e.g. farming drills, arenas with only a few players
        self.active = torch.ones(B, P, dtype=torch.bool, device=d)
        # per-slot spawn mass range (lo, hi; log-uniform), 0 = the default/curriculum: e.g. hunting
        # drills where learners start big among small prey
        self.spawn_mass = torch.zeros(B, P, 2, device=d)
        self.l_kills = torch.zeros(B, NL, device=d)
        self.whale_resets = 0

    def set_reward(self, arenas, virus_reward=(0.0, 0.0, 0.0), explore_reward=0.0, kill_reward=(0.0, 0.0),
                   growth: dict | None = None, action_bonus=(0.0, 0.0)):
        """Reward weights for `arenas` (a slice over B): virus_reward (shot, pop, popmass),
        explore_reward per new map square, kill_reward (per enemy cell eaten, per 200 enemy mass
        eaten), action_bonus (per split, per eject the game carries out).
        growth: log-mass change instead of linear mass, minus a "par" growth rate while above
        par_mass. Farming at a steady absolute pace earns less and less in log terms as the cell
        grows, and falls below par: only accelerating growth (eating players) keeps paying.
        dict(par_doubling_min=4.0, par_mass=500.0, death=1.0); None = linear mass reward."""
        vec = lambda xs: torch.tensor([float(x) for x in xs], device=self.dev)[:, None, None]
        self.r_virus[:, arenas] = vec(virus_reward)
        self.r_kill[:, arenas] = vec(kill_reward)
        self.r_explore[arenas] = float(explore_reward)
        self.r_bonus[:, arenas] = vec(action_bonus)
        self.r_growth[arenas] = bool(growth)
        if growth:
            par_step = math.log(2) / (growth.get("par_doubling_min", 4.0) * 60 / 0.08)
            self.r_par[:, arenas] = vec([par_step, growth.get("par_mass", 500.0), growth.get("death", 1.0)])
        # python flags: skip the work no arena asks for
        self._any_explore = bool((self.r_explore > 0).any())
        self._any_bonus = bool((self.r_bonus > 0).any())
        self._any_growth = bool(self.r_growth.any())

    def set_tracked(self, mask: torch.Tensor):
        """mask [B, NL] bool: slots whose lives enter pop_stats() (e.g. only the learners)."""
        self.track = mask.to(self.dev)

    def _clear_lives(self, m: torch.Tensor):
        for t in (self.l_peak, self.l_steps, self.l_shots, self.l_pops, self.l_stuck, self.l_ret, self.l_kills):
            t.masked_fill_(m, 0)
        self.seen.masked_fill_(m[..., None], False)

    # ------------------------------------------------------------------ helpers
    def _observe(self):
        grid, scal, frame, alive, mass = self.enc.encode(self.obs_dtype)
        self.frame = frame
        return grid, scal, alive, mass

    def _drop_inactive(self, arenas: torch.Tensor | None = None):
        sel = torch.ones(self.B, 1, dtype=torch.bool, device=self.dev) if arenas is None else arenas[:, None]
        if not bool(self.active.all()):
            self.w.calive.masked_fill_(~self.active.repeat_interleave(self.w.K, dim=1) & sel, False)
        if bool((self.spawn_mass[..., 0] > 0).any()):   # fresh arenas: drill sizes from the start
            set_ = (self.spawn_mass[..., 0] > 0) & sel
            r2 = self._drill_size() ** 2
            v = self.w.cr2.view(self.B, self.P, self.w.K)
            v[..., 0] = torch.where(set_ & (v[..., 0] > 0), r2, v[..., 0])

    def _drill_size(self) -> torch.Tensor:
        lo, hi = self.spawn_mass[..., 0].clamp(min=1), self.spawn_mass[..., 1].clamp(min=1)
        u = torch.rand(self.B, self.P, generator=self.gen, device=self.dev)
        return torch.floor(torch.sqrt(100 * torch.exp(lo.log() + (hi.log() - lo.log()) * u)))

    def _alive_mass(self):
        """[B,NL] alive and mass as the client sees it (sum of floor(size)^2 / 100)."""
        w = self.w
        fs = torch.floor(w.cr2.sqrt()) * w.calive
        mass = (fs * fs / 100).view(self.B, self.P, w.K).sum(-1)[:, :self.NL]
        return w.alive_players()[:, :self.NL], mass

    def reset_arenas(self, arenas: torch.Tensor):
        """Restart only `arenas` ([B] bool), e.g. after their players' roles changed. Their lives
        in progress are dropped from the statistics; observations refresh at the next step."""
        self.w.reset(arenas)
        self._drop_inactive(arenas)
        m = arenas[:, None].expand(self.B, self.NL)
        self.enc.reset_agents(m)
        self._clear_lives(m)
        self.prev_vstat.masked_fill_(m[None], 0)
        self.prev_kstat.masked_fill_(m[None], 0)
        self.prev_alive, self.prev_mass = self._alive_mass()

    def reset(self):
        self.w.reset()
        self._drop_inactive()
        ones = torch.ones(self.B, self.NL, dtype=torch.bool, device=self.dev)
        self.enc.reset_agents(ones)
        self.py_steps = 0
        self._clear_lives(ones)
        self.prev_vstat.zero_()
        self.prev_kstat.zero_()
        grid, scal, alive, mass = self._observe()
        self.prev_alive, self.prev_mass = self._alive_mass()
        z = torch.zeros(self.n_agents, device=self.dev)
        return grid, scal, z, torch.zeros(self.n_agents, dtype=torch.bool, device=self.dev), alive.reshape(-1)

    def _mouse(self, actions: torch.Tensor) -> torch.Tensor:
        """rl.actions.decode: heading -> world target at distance ext from the last frame's centre."""
        cx, cy, ext = self.frame
        move = actions[:, 0].view(self.B, self.NL)
        a = 2 * math.pi * move.clamp(max=DIRS).float() / DIRS
        stay = move >= DIRS
        tx = cx + torch.where(stay, torch.zeros_like(cx), torch.cos(a) * ext)
        ty = cy + torch.where(stay, torch.zeros_like(cy), torch.sin(a) * ext)
        return torch.stack([tx, ty], -1)

    def pop_stats(self, arenas=slice(None)) -> dict:
        """Per-life statistics of the lives finished in `arenas` since the last call (one host sync)."""
        t = torch.cat([self._tot[:, arenas].sum(1), self._pmax[arenas].max()[None]]).tolist()  # t[9] kills, t[10] peak max
        self._tot[:, arenas] = 0
        self._pmax[arenas] = 0
        n = max(t[0], 1.0)
        return {"lives": t[0], "died_share": t[1] / n, "peak": t[2] / n, "peak_max": t[10], "kills": t[9] / n, "steps": t[3] / n,
                "shots": t[4] / n, "pops": t[5] / n, "stuck_share": t[6] / n, "ret": t[7] / n,
                "lives_over_10k": t[8]}

    def _finish(self, ended: torch.Tensor, died: torch.Tensor):
        """Add the lives that ended this step ([B,NL] bool) to the running statistics and restart them."""
        e = (ended & self.track).float()
        share = self.l_stuck / self.l_steps.clamp(min=1)
        self._tot += torch.stack([e.sum(1), (e * died.float()).sum(1), (e * self.l_peak).sum(1),
                                  (e * self.l_steps).sum(1), (e * self.l_shots).sum(1), (e * self.l_pops).sum(1),
                                  (e * share).sum(1), (e * self.l_ret).sum(1),
                                  (e * (self.l_peak >= 10000).float()).sum(1), (e * self.l_kills).sum(1)])
        self._pmax = torch.maximum(self._pmax, (e * self.l_peak).amax(1))
        self._clear_lives(ended)

    @torch.no_grad()
    def step(self, actions, observe: bool = True):
        w, B, P, NL, dev = self.w, self.B, self.P, self.NL, self.dev
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        actions = actions.to(dev).long().view(B * NL, 3)
        mouse = torch.zeros(B, P, 2, device=dev)
        split = torch.zeros(B, P, dtype=torch.bool, device=dev)
        eject = torch.zeros(B, P, dtype=torch.bool, device=dev)
        mouse[:, :NL] = self._mouse(actions)
        split[:, :NL] = actions[:, 1].view(B, NL) > 0
        eject[:, :NL] = actions[:, 2].view(B, NL) > 0
        if NL < P:  # scripted bots aim at a far point along their heading
            ba = scripted_actions(w, self.bot_split_prob)[:, NL:]
            cx = (w.cx * w.cr2).view(B, P, w.K).sum(-1)[:, NL:] / (w.cr2 * w.calive).view(B, P, w.K).sum(-1)[:, NL:].clamp(min=1)
            cy = (w.cy * w.cr2).view(B, P, w.K).sum(-1)[:, NL:] / (w.cr2 * w.calive).view(B, P, w.K).sum(-1)[:, NL:].clamp(min=1)
            ang = 2 * math.pi * ba[..., 0].clamp(max=DIRS).float() / DIRS
            far = torch.where(ba[..., 0] >= DIRS, torch.zeros_like(cx), torch.full_like(cx, 1000.0))
            mouse[:, NL:, 0] = cx + torch.cos(ang) * far
            mouse[:, NL:, 1] = cy + torch.sin(ang) * far
            split[:, NL:] = ba[..., 1] > 0
        was_alive = w.alive_players()
        if self._any_bonus:                                # what the game will carry out: split needs a
            r2v = w.cr2.view(B, P, w.K)                       # piece >= 35 mass and a free cell slot
            can = (r2v.amax(-1) >= MIN_SPLIT_R2) & (w.calive.view(B, P, w.K).sum(-1) < w.K)
            self._acted = torch.stack([split & can & was_alive, eject & can & was_alive])[:, :, :NL].float()
        # respawn size: learners sometimes start big (curriculum), everyone else at the default
        size = torch.full((B, P), math.sqrt(START_R2), device=dev)
        if self.curriculum_p > 0:
            lo, hi = self.curriculum_mass
            big = torch.rand(B, NL, generator=self.gen, device=dev) < self.curriculum_p
            mass = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(B, NL, generator=self.gen, device=dev))
            size[:, :NL] = torch.where(big, torch.floor(torch.sqrt(100 * mass)), size[:, :NL])
        drill = self.spawn_mass[..., 0] > 0
        if bool(drill.any()):
            size = torch.where(drill, self._drill_size(), size)
        w.step(mouse, split, eject, was_alive, respawn=~was_alive & self.active, spawn_size=size)

        # ---- reward before any episode reset
        alive_now, mass_now = self._alive_mass()
        pa, pm = self.prev_alive, self.prev_mass
        died = pa & ~alive_now
        rew = torch.where(alive_now & pa, (mass_now - pm) * REWARD_SCALE, torch.zeros_like(mass_now))
        rew = torch.where(died, -pm * REWARD_SCALE, rew)
        if self._any_growth:
            lg = lambda m: torch.log(m.clamp(min=SPAWN_MASS))
            par = self.r_par[0] * (mass_now > self.r_par[1]).float()
            rg = torch.where(alive_now & pa, lg(mass_now) - lg(pm) - par, torch.zeros_like(mass_now))
            rg = torch.where(died, -(lg(pm) - math.log(SPAWN_MASS)) - self.r_par[2], rg)
            rew = torch.where(self.r_growth, rg, rew)
        if self._any_bonus:
            rew = rew + self.r_bonus[0] * self._acted[0] + self.r_bonus[1] * self._acted[1]
        done = died.clone()
        live = alive_now | pa
        # virus shots / pops of this step
        vs = w.vstat[:, :, :NL].clone()
        dv = torch.where(live[None], vs - self.prev_vstat, torch.zeros_like(vs))
        self.prev_vstat = vs
        rew = rew + self.r_virus[0] * dv[0] + self.r_virus[1] * dv[1] + self.r_virus[2] * dv[2] / 200
        ks = w.kstat[:, :, :NL].clone()
        dk = torch.where(live[None], ks - self.prev_kstat, torch.zeros_like(ks))
        self.prev_kstat = ks
        rew = rew + self.r_kill[0] * dk[0] + self.r_kill[1] * dk[1] / 200
        # main (largest) cell position: novelty grid and stuck detection
        r2 = w.cr2.view(B, P, w.K)[:, :NL]
        mi = r2.argmax(-1, keepdim=True)
        mx = torch.trunc(w.cx.view(B, P, w.K)[:, :NL].gather(-1, mi)[..., 0])
        my = torch.trunc(w.cy.view(B, P, w.K)[:, :NL].gather(-1, mi)[..., 0])
        steps_now = torch.where(alive_now, self.l_steps + 1, torch.zeros_like(self.l_steps))
        if self._any_explore:
            gi = ((mx + w.map / 2) // EXPLORE_CELL).clamp(0, self.ng - 1).long()
            gj = ((my + w.map / 2) // EXPLORE_CELL).clamp(0, self.ng - 1).long()
            oh = (gi * self.ng + gj)[..., None] == torch.arange(self.ng * self.ng, device=dev)
            new = alive_now & ~(oh & self.seen).any(-1)
            self.seen |= oh & alive_now[..., None]
            rew = rew + self.r_explore * (new & (steps_now > 1)).float()
        t = self.py_steps
        cur = torch.stack([mx, my], -1)
        old = self._trail[(t + 1) % STUCK_WINDOW]            # position 24 steps ago
        stuck = alive_now & (steps_now > STUCK_WINDOW) & ((cur - old).norm(dim=-1) < STUCK_DIST)
        self._trail[t % STUCK_WINDOW] = cur
        # per-life accumulators
        self.l_peak = torch.where(alive_now, torch.maximum(self.l_peak, mass_now), torch.maximum(self.l_peak, pm))
        self.l_steps = torch.where(alive_now, steps_now, self.l_steps)   # a life ending now keeps its length
        self.l_ret = self.l_ret + torch.where(alive_now, torch.log(mass_now.clamp(min=1.0) / SPAWN_MASS) * 0.001,
                                              torch.zeros_like(mass_now))
        self.l_shots = self.l_shots + dv[0]
        self.l_pops = self.l_pops + dv[1]
        self.l_kills = self.l_kills + dk[0]
        self.l_stuck = self.l_stuck + stuck.float()
        self.enc.reset_agents(died)
        self._finish(died, died)
        # episode truncation: lives in progress count as finished, the arenas restart
        self.py_steps += 1
        if self.py_steps >= self.episode_steps:      # all arenas run in lockstep, so they truncate together
            done = done | alive_now
            self._finish(alive_now, torch.zeros_like(alive_now))
            w.reset()
            self._drop_inactive()
            ones = torch.ones(B, NL, dtype=torch.bool, device=dev)
            self.enc.reset_agents(ones)
            self.py_steps = 0
            self._clear_lives(ones)
            self.prev_vstat.zero_()
            self.prev_kstat.zero_()
        self.prev_alive, self.prev_mass = self._alive_mass()
        if self.max_player_mass and self.py_steps % 25 == 0:
            whale = (self.w.total_mass() > self.max_player_mass).any(1)            # [B]
            if bool(whale.any()):
                cut = whale[:, None] & self.prev_alive
                done = done | cut
                self._finish(cut, torch.zeros_like(cut))
                self.reset_arenas(whale)
                self.whale_resets += int(whale.sum())
        if not observe:
            return None, None, rew.reshape(-1), done.reshape(-1), self.prev_alive.reshape(-1)
        grid, scal, alive, mass = self._observe()
        return grid, scal, rew.reshape(-1), done.reshape(-1), alive.reshape(-1)
