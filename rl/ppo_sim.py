"""PPO self-play fully on the GPU, on the batched simulator (sim/).

    uv run python -m rl.ppo_sim --run s1 [--init runs/rl/g8_048.pt] [--b 256] [--steps 32] [--skip 2]

B arenas x P players step on the GPU (`sim.env.SimEnv`); observations, rollouts and PPO all stay
on the GPU. Slot roles: in most arenas every slot is a learner playing the current policy
(self-play). A fraction of arenas gives half of the slots to opponents instead: `--league-frac`
arenas use frozen snapshots of this run (a league: the starting policy plus a snapshot every
`--league-every` iterations), `--bot-frac` arenas use the scripted bot (sim.bots). Drill arenas
(farm, hunt, flee, giant) come first. Only learner slots produce training samples. Lives'
statistics (peak mass, shots, pops, ...) are logged for the learner slots only.

Frame skip (`--skip k`): every policy decides every k env steps (k x 80 ms) and holds its move in
between; a split or eject fires once, on the decision's first step. Rewards are summed over the
k steps and gamma applies per decision (gamma ** k), so `--gamma` keeps its per-step meaning.

The training loop is the `Learner` class (one network, its arenas, its league, its PPO state), so
several learners can share one env: rl.pbt runs a whole population in one process this way.

Network: rl.model.Policy with the sim layout (c=15: the guided net's 13 channels + killable prey +
enemy mass density; s=16, g=48) and heads (17, 2, 2): 16 headings + stay, split, eject. There is
no FOLLOW action (a net trained here has no champion). `--init` takes a guided checkpoint (its
FOLLOW logit row is dropped, its hint input weights stay but see zeros) or an earlier sim run.

Checkpoints: runs/rl/<run>/ckpt_NNNNN.pt and latest.pt, written with rl.common.save and tagged
`sim=True, hint=False, skip=k`; rl.policy.NNPolicy plays them on the real server.
"""

import argparse
import copy
import json
import statistics
import time
import weakref
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sim.bots import farmer_actions, scripted_actions
from sim.env import SimEnv

from .common import ROOT, load, save
from .model import Policy

C_IN, S_IN, G_IN = 15, 16, 48
ARCH_KEYS = ("width", "convs", "pool", "in_mask", "s_mask", "shared", "coords")
HEADS = (17, 2, 2)
ROLE_LEARN, ROLE_LEAGUE, ROLE_BOT, ROLE_OFF = 0, 1, 2, 3


ENTITY_KEYS = ("kind", "d", "layers", "heads", "width")


def new_model(width: int = 256, **arch) -> Policy:
    if arch.get("kind") == "entity":       # transformer over object tokens (rl.entity_model, sim/entities.py)
        from sim.entities import FD, N_TOK
        from .entity_model import EntityPolicy
        return EntityPolicy(n_tok=N_TOK, fd=FD, s=S_IN, width=width, **{k: v for k, v in arch.items() if k != "kind"})
    return Policy(width=width, c=C_IN, s=S_IN, heads=HEADS, follow=None, g=G_IN, **arch)


def arch_of(model: Policy) -> dict:
    if model.config.get("kind") == "entity":
        return {k: model.config[k] for k in ENTITY_KEYS}
    return {k: model.config[k] for k in ARCH_KEYS if k in model.config}


def obs_kind(model) -> str:
    return getattr(model, "obs", "grid")


def obs_for(model, env, g):
    """The observation `model` reads: the env's grid (g) or its entity tokens."""
    if obs_kind(model) == "entity" and env.obs == "both":
        return env.ent
    return g


def from_checkpoint(path: str, dev: str) -> Policy:
    """A sim-layout policy initialised from a guided (18 moves, FOLLOW) or sim checkpoint.
    Nets from before the extra input channels (c=13) get zero weights for them, so they play
    exactly as before and can learn to use them."""
    src = load(path, dev)
    cfg = src.config
    if cfg.get("kind") == "entity":   # the token count may have grown (e.g. food-cluster tokens): weights are per token
        from sim.entities import FD, N_TOK
        if cfg["fd"] < FD:            # new token features (e.g. the commitment flag): zero weights, same play
            new = new_model(**arch_of(src)).to(dev)
            sd = src.state_dict()
            for k in ("enc.tok.weight", "critic.tok.weight"):
                w = sd[k]
                sd[k] = torch.cat([w, w.new_zeros(w.shape[0], FD - w.shape[1])], 1)
            if cfg["fd"] < 21:        # nets from before the commitment inputs: scalars 8-10 were always
                for k in ("enc.state.weight", "critic.state.weight"):        # zero, untrained weights
                    sd[k] = sd[k].clone()
                    sd[k][:, 8:11] = 0
            new.load_state_dict(sd)
            new.skip = getattr(src, "skip", 1)
            src = new
        src.n_tok = N_TOK
        src.config.update(n_tok=N_TOK, heads_out=(17 + N_TOK, 2, 2))
        return src
    if cfg["c"] not in (13, C_IN) or cfg["s"] != S_IN:
        raise SystemExit(f"{path}: needs c=13/{C_IN} s=16, got c={cfg['c']} s={cfg['s']}")
    model = new_model(**arch_of(src)).to(dev)
    sd = src.state_dict()
    n = HEADS[0]
    sd["heads.0.weight"], sd["heads.0.bias"] = sd["heads.0.weight"][:n], sd["heads.0.bias"][:n]
    if cfg["c"] < C_IN and cfg.get("in_mask") is None:
        for key in ("conv.0.weight", "critic_conv.0.weight"):
            if key in sd:
                w = sd[key]
                sd[key] = torch.cat([w, w.new_zeros(w.shape[0], C_IN - w.shape[1], *w.shape[2:])], 1)
    model.load_state_dict(sd)
    model.skip = getattr(src, "skip", 1)
    return model


def frozen(m: Policy) -> Policy:
    m = copy.deepcopy(m).eval()
    m.binary_eps = 0.0
    for q in m.parameters():
        q.requires_grad_(False)
    return m


AMP = True   # bfloat16 network passes (see Policy.dists); --no-amp turns it off
torch.distributions.Distribution.set_default_validate_args(False)   # the checks sync the GPU


class GraphedAct:
    """model.act for one fixed batch size as a CUDA graph: one launch instead of dozens of small
    kernels from Python (acting was launch-bound). The graph reads the model's parameters in
    place, so it stays valid while the optimizer updates them; RNG is graph-safe."""

    def __init__(self, model: Policy, g: torch.Tensor, s: torch.Tensor):
        self.g, self.s = g.clone(), s.clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(3):                                     # warm-up: cudnn tuning, allocator
                model.act(self.g, self.s, amp=AMP)
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = model.act(self.g, self.s, amp=AMP)

    def __call__(self, g, s):
        self.g.copy_(g)
        self.s.copy_(s)
        self.graph.replay()
        return tuple(o.clone() for o in self.out)


_graphs: "OrderedDict[tuple, GraphedAct]" = OrderedDict()   # (model id, batch, eps) -> graph, LRU
_graph_owner = weakref.WeakValueDictionary()                 # model id -> model (drop graphs of dead models)
MAX_GRAPHS = 32   # each keeps its own activation memory; league snapshots come and go


def act_sampled(model: Policy, g: torch.Tensor, s: torch.Tensor):
    """Sample actions for a batch; on CUDA with AMP it replays a CUDA graph (`GraphedAct`)."""
    if not AMP or not g.is_cuda or len(g) == 0:
        with torch.no_grad():
            return model.act(g if AMP else g.float(), s, amp=AMP)
    key = (id(model), len(g), model.binary_eps)
    if _graph_owner.get(id(model)) is not model:      # a new model reusing a dead one's id
        for k in [k for k in _graphs if k[0] == id(model)]:
            del _graphs[k]
        _graph_owner[id(model)] = model
    if key not in _graphs:
        while len(_graphs) >= MAX_GRAPHS:
            _graphs.popitem(last=False)
        _graphs[key] = GraphedAct(model, g, s)
    _graphs.move_to_end(key)
    return _graphs[key](g, s)


# ---- arena layout ------------------------------------------------------------------------------
LAYOUT_DEFAULTS = dict(league_frac=0.2, bot_frac=0.1, farm_frac=0.0, farm_players=8, hunt_frac=0.0,
                       hunt_players=10, flee_frac=0.0, flee_players=20, flee_hunters=10, giant_frac=0.0,
                       giants=4, giant_learners=2, farm_spawn=None, hunt_spawn=(400.0, 2000.0),
                       prey_spawn=(20.0, 200.0), prey_nets=0.0)


def layout(env: SimEnv, lo: int, nb: int, cfg: dict) -> torch.Tensor:
    """Roles [nb, P] for arenas lo..lo+nb (0 learner, 1 league, 2 scripted bot, 3 off), and their
    drill settings in `env` (active slots, spawn sizes). Drill arenas come first:
      farm:  only `farm_players` learners, nothing else: a clean farming signal;
      hunt:  `hunt_players` learners at 400-2000 mass among small scripted bots (20-200);
      flee:  `flee_players` learners at 100-600 mass with `flee_hunters` scripted bots at 1500-6000;
      giant: a full arena of learners plus `giants` scripted bots at 5k-20k mass (like a strong
             human on the real server) and `giant_learners` learners that start that big.
    Then `league_frac` arenas with half the slots league snapshots, `bot_frac` with half bots.
    Drill players respawn at their drill size, so the pressure lasts all game."""
    c = {**LAYOUT_DEFAULTS, **cfg}
    P, dev = env.P, env.dev
    n_league, n_bot = round(nb * c["league_frac"]), round(nb * c["bot_frac"])
    n_farm, n_hunt = round(nb * c["farm_frac"]), round(nb * c["hunt_frac"])
    n_flee, n_giant = round(nb * c["flee_frac"]), round(nb * c["giant_frac"])
    n_drill = n_farm + n_hunt + n_flee + n_giant
    assert n_drill + n_bot <= nb, "drill + bot arenas exceed the arenas"
    n_league = max(0, min(n_league, nb - n_drill - n_bot))
    env.active[lo:lo + nb] = True
    env.spawn_mass[lo:lo + nb] = 0
    role = torch.zeros(nb, P, dtype=torch.long, device=dev)
    half = P - P // 2
    role[nb - n_league - n_bot:nb - n_bot, half:] = ROLE_LEAGUE
    role[nb - n_bot:, half:] = ROLE_BOT
    t = lambda a, b: torch.tensor([a, b], device=dev)
    a = 0
    for i in range(a, a + n_farm):
        env.active[lo + i, c["farm_players"]:] = False
        role[i, c["farm_players"]:] = ROLE_OFF
        if c["farm_spawn"]:      # farmers (re)start at this mass range: every size gets practice
            env.spawn_mass[lo + i, :c["farm_players"]] = t(*map(float, c["farm_spawn"]))
    a += n_farm
    for i in range(a, a + n_hunt):
        k = c["hunt_players"]
        role[i, k:] = ROLE_BOT
        n_net = round((P - k) * c["prey_nets"])    # prey played by frozen nets (league / hall of fame): they
        role[i, P - n_net:] = ROLE_LEAGUE          # don't flee, so eating others pays from the first meeting
        env.spawn_mass[lo + i, :k] = t(*map(float, c["hunt_spawn"]))
        env.spawn_mass[lo + i, k:] = t(*map(float, c["prey_spawn"]))
    a += n_hunt
    for i in range(a, a + n_flee):
        k, hn = c["flee_players"], c["flee_hunters"]
        role[i, k:k + hn] = ROLE_BOT
        role[i, k + hn:] = ROLE_OFF
        env.active[lo + i, k + hn:] = False
        env.spawn_mass[lo + i, :k] = t(100.0, 600.0)
        env.spawn_mass[lo + i, k:k + hn] = t(1500.0, 6000.0)
    a += n_flee
    for i in range(a, a + n_giant):
        gb, gl = c["giants"], c["giant_learners"]
        role[i, P - gb:] = ROLE_BOT
        env.spawn_mass[lo + i, P - gb:] = t(5000.0, 20000.0)
        env.spawn_mass[lo + i, :gl] = t(5000.0, 20000.0)
    return role


def parse_growth(x):
    if not x:
        return None
    if isinstance(x, str):
        x = [float(v) for v in x.split(",")]
    return dict(zip(("par_doubling_min", "par_mass", "death"), map(float, x)))


# ---- one learning network --------------------------------------------------------------------------
class Learner:
    """One network learning in arenas lo..lo+nb of a shared env: its learner slots, its league
    (own snapshots + hall of fame), its rollout buffers and PPO update, its logs and checkpoints.

    hp: lr, ent, gamma, lam, clip, epochs, mb, target_kl, vf, vf_warmup, explore, league_every,
    league_size, league_models, save_every, plus the arena layout keys (see `layout`) and the
    reward keys (virus_reward, explore_reward, kill_reward, growth, action_bonus)."""

    def __init__(self, name: str, model: Policy, env: SimEnv, lo: int, nb: int, hp: dict, out: Path,
                 hof: list[Policy] | None = None, teacher: Policy | None = None, distill_iters: int = 0,
                 start_it: int = 0, log=None, skip: int = 1, resume_league: bool = True):
        self.name, self.model, self.env, self.lo, self.nb = name, model, env, lo, nb
        self.hp, self.out, self.skip = hp, Path(out), skip
        self.teacher, self.distill_until = teacher, start_it + distill_iters
        # imitate the nearest-pellet farmer (sim.bots.farmer_actions) first: DAgger on our own states
        self.bc_until = start_it + hp.get("bc_farmer_iters", 0)
        # then the imitation term fades out linearly over bc_fade_iters while the mass reward takes over
        self.fade_until = self.bc_until + (hp.get("bc_fade_iters", 0) if hp.get("bc_farmer_iters", 0) else 0)
        self.it = self.start_it = start_it
        self.log_f = log
        dev = self.dev = env.dev
        P = env.P
        model.binary_eps = hp.get("explore", 0.02)
        # where the split / eject exploration floor applies (farming: W only wastes mass, and a
        # split can only pay for a big cell)
        model.split_explore_min_mass = hp.get("split_explore_min_mass", 70.0)
        model.eject_explore = hp.get("eject_explore", True)
        self.opt = torch.optim.Adam(model.parameters(), lr=hp.get("lr", 1e-4), eps=1e-5, fused=True)
        env.set_reward(slice(lo, lo + nb), hp.get("virus_reward", (0, 0, 0)), hp.get("explore_reward", 0.0),
                       hp.get("kill_reward", (0, 0)), parse_growth(hp.get("growth")),
                       hp.get("action_bonus", (0, 0)), hp.get("death_scale", 1.0))
        role = layout(env, lo, nb, hp)
        flat = role.reshape(-1)
        base = lo * P
        self.learn_idx = base + (flat == ROLE_LEARN).nonzero()[:, 0]
        self.bot_idx = base + (flat == ROLE_BOT).nonzero()[:, 0]
        self.N = len(self.learn_idx)
        track = env.track
        track[lo:lo + nb] = role == ROLE_LEARN
        env.set_tracked(track)
        # league: own snapshots (resumed from disk) + hall of fame, split over a few groups of arenas
        self.snaps = [frozen(model)]
        old = sorted((self.out / "league").glob("it*.pt")) if resume_league else []
        if old:
            keep = old[:1] + old[1:][-(hp.get("league_size", 8) - 1):]
            self.snaps = [frozen(from_checkpoint(str(f), dev)) for f in keep]
        self.hof = hof or []
        league_arenas = [a for a in range(nb) if bool((role[a] == ROLE_LEAGUE).any())]
        self.league_idx, self.league_models = [], []
        if league_arenas:
            k = min(hp.get("league_models", 4), len(league_arenas))
            for ch in np.array_split(np.array(league_arenas), k):
                cht = torch.as_tensor(ch, device=dev)
                ix = (role[cht] == ROLE_LEAGUE).nonzero()
                self.league_idx.append(base + cht[ix[:, 0]] * P + ix[:, 1])
                self.league_models.append(None)
        self.assign_league()
        self.tb = SummaryWriter(str(ROOT / "logs" / "tb" / name))
        if not (self.out / "league" / "it00000.pt").exists():
            self.save(self.out / "league" / "it00000.pt")
        self.t_last = time.time()

    @property
    def arenas(self) -> slice:
        return slice(self.lo, self.lo + self.nb)

    def assign_league(self):
        # league_pool "hof": only the hall of fame plays the league slots (e.g. prey nets for hunters)
        pool = self.hof if self.hp.get("league_pool") == "hof" and self.hof else self.snaps + self.hof
        for i in range(len(self.league_models)):
            self.league_models[i] = pool[np.random.randint(len(pool))]

    def save(self, path: Path):
        save(self.model, path, it=self.it, sim=True, hint=False, skip=self.skip)

    def print(self, msg: str):
        print(msg, file=self.log_f, flush=True)

    # ---- rollout
    def begin(self, T: int, gdim):
        """Allocate (or reuse) the T-step rollout buffers before a collection."""
        N, dev = self.N, self.dev
        if getattr(self, "T", None) != T:      # buffers live across iterations (no allocator churn)
            self.T = T
            self.G = torch.empty((T, N, *gdim), dtype=torch.float16, device=dev)
            self.Sc = torch.empty((T, N, S_IN), device=dev)
            self.Act = torch.zeros((T, N, 3), dtype=torch.long, device=dev)
            self.Logp = torch.zeros((T, N), device=dev)
            self.Val = torch.zeros((T + 1, N), device=dev)
            self.Rew = torch.zeros((T, N), device=dev)
            self.Done = torch.zeros((T, N), device=dev)
            self.Trunc = torch.zeros((T, N), device=dev)
            self.Valid = torch.zeros((T + 1, N), device=dev)
        if self.it < self.fade_until and getattr(self, "TAct", None) is None:
            self.TAct = torch.zeros((T, N, 3), dtype=torch.long, device=dev)
        elif self.it >= self.fade_until:
            self.TAct = None
        lr = self.hp.get("bc_lr", 1e-3) if self.it < self.bc_until else self.hp.get("lr", 1e-4)
        for pg_ in self.opt.param_groups:
            pg_["lr"] = lr
        self.splits = self.ejects = torch.zeros((), device=dev)
        self.t0 = time.time()

    def act(self, t: int, g, s, v, acts):
        """Sample the learners' actions (stored for training) and the league's; write them to acts."""
        ix = self.learn_idx
        gl, sl = obs_for(self.model, self.env, g)[ix], s[ix]
        a, logp, val = act_sampled(self.model, gl, sl)
        self.G[t], self.Sc[t], self.Act[t], self.Logp[t], self.Val[t] = gl, sl, a, logp, val
        vl = v[ix]
        self.Valid[t] = vl.float()
        self.splits = self.splits + (a[:, 1] * vl).sum()
        if self.it < self.fade_until:
            ta = farmer_actions(self.env)[ix]
            if obs_kind(self.model) == "entity":     # the same target as a pointer: the nearest pellet token
                k0 = 16 + self.env.ent_enc.k["cell"]
                near = self.env.ent[ix, k0, 0] > 0.5
                ta[:, 0] = torch.where(near, torch.full_like(ta[:, 0], 17 + k0), ta[:, 0])
            self.TAct[t] = ta
        self.ejects = self.ejects + (a[:, 2] * vl).sum()
        acts[ix] = a
        for m, lx in zip(self.league_models, self.league_idx):
            acts[lx] = act_sampled(m, obs_for(m, self.env, g)[lx], s[lx])[0]

    def record(self, t: int, r, d, tr=None):
        self.Rew[t] = r[self.learn_idx]
        self.Done[t] = d[self.learn_idx].float()
        self.Trunc[t] = tr[self.learn_idx].float() if tr is not None else 0.0

    def finish(self, g, s, v):
        ix = self.learn_idx
        self.Val[self.T] = act_sampled(self.model, obs_for(self.model, self.env, g)[ix], s[ix])[2]
        self.Valid[self.T] = v[ix].float()
        self.collect_s = time.time() - self.t0

    # ---- PPO
    def update(self):
        """One PPO iteration on the finished rollout: GAE, then `epochs` of minibatch updates (plus the
        distillation / farmer-imitation terms while active); saves and snapshots on schedule."""
        hp, model, T, N, dev = self.hp, self.model, self.T, self.N, self.dev
        self.it += 1
        it = self.it
        t1 = time.time()
        gamma, lam = hp.get("gamma", 0.998) ** self.skip, hp.get("lam", 0.95)
        clip, mb = hp.get("clip", 0.2), hp.get("mb", 2048)
        adv = torch.zeros((T, N), device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            # a life cut short by a restart (not a death) keeps its value: bootstrap from V(s_t), the
            # best estimate we have; the next state belongs to the restarted arena, so the chain stops
            tr = self.Trunc[t]
            cont = (1 - self.Done[t]) * (1 - tr) * self.Valid[t + 1]
            delta = self.Rew[t] + gamma * (self.Val[t + 1] * cont + tr * self.Val[t]) - self.Val[t]
            last = delta + gamma * lam * cont * last
            adv[t] = last * self.Valid[t]
        ret = adv + self.Val[:T]
        ids = (self.Valid[:T].reshape(-1) > 0).nonzero()[:, 0]
        n = len(ids)
        gdim = self.G.shape[2:]
        Gf, Sf, Af = self.G.view(T * N, *gdim), self.Sc.view(T * N, -1), self.Act.view(T * N, 3)
        Lf, advf, retf = self.Logp.view(-1), adv.view(-1), ret.view(-1)
        a_ = advf[ids]
        advn = torch.zeros_like(advf)
        advn[ids] = (a_ - a_.mean()) / (a_.std() + 1e-8)
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "maxlogr": []}
        nmb = max(n // mb, 1)
        distill = self.teacher is not None and it <= self.distill_until
        bc = it <= self.bc_until
        fade = 0.0 if bc or it > self.fade_until else 1 - (it - self.bc_until) / max(self.fade_until - self.bc_until, 1)
        TAf = self.TAct.view(T * N, 3) if bc or fade > 0 else None
        for ep in range(hp.get("epochs", 3)):
            # early stop on the share of clipped samples: the KL mean was driven by a few outliers
            # (a near-deterministic policy) and stopped most updates after one pass
            if not bc and stats["clip"] and statistics.fmean(stats["clip"][-nmb:]) > hp.get("clip_stop", 0.3):
                break
            perm = ids[torch.randperm(n, device=dev)]
            # full minibatches only: a new batch size makes cudnn.benchmark re-tune every conv
            # (seconds of CPU per update); the remainder sits this epoch out (a random few)
            for k in range(0, max(n - n % mb, 1), mb):
                ix = perm[k:k + mb]
                gx = Gf[ix] if AMP else Gf[ix].float()
                dists, value = model.dists(gx, Sf[ix], amp=AMP)
                logp, ent = model.logp_entropy(dists, Af[ix])
                ent = ent.mean()
                logr = (logp - Lf[ix]).clamp(-20.0, 20.0)   # bounded: an exp() overflow / log(0) poisons the step
                ratio = logr.exp()
                A_ = advn[ix]
                surr = torch.min(ratio * A_, ratio.clamp(1 - clip, 1 + clip) * A_)
                # dual clip: with A < 0 a huge ratio (an action the floor made rare, now likely) made
                # r * A unbounded and one sample could steer the whole step (a member froze that way)
                surr = torch.where(A_ < 0, torch.max(surr, 3.0 * A_), surr)
                pg = -surr.mean()
                vf = 0.5 * (value - retf[ix]).pow(2).mean()
                if bc:        # imitate the farmer's move only: split / eject are left to the reward to discover
                    loss = -dists[0].log_prob(TAf[ix][:, 0]).mean() + hp.get("vf", 0.5) * vf
                elif distill:   # new architecture: imitate the parent (DAgger: our states, its actions)
                    with torch.no_grad():
                        tdists, _ = self.teacher.dists(gx, Sf[ix], amp=AMP)
                    kl_t = sum(torch.distributions.kl_divergence(td, sd_).mean() for td, sd_ in zip(tdists, dists))
                    loss = kl_t + hp.get("vf", 0.5) * vf
                elif it <= self.start_it + hp.get("vf_warmup", 0):
                    loss = hp.get("vf", 0.5) * vf
                else:
                    loss = pg + hp.get("vf", 0.5) * vf - hp.get("ent", 0.001) * ent
                    if fade > 0:   # fading imitation of the move
                        loss = loss - fade * dists[0].log_prob(TAf[ix][:, 0]).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.detach()); stats["vf"].append(vf.detach()); stats["ent"].append(ent.detach())
                    stats["kl"].append(((ratio - 1) - logr).mean())
                    stats["maxlogr"].append(logr.abs().max())
                    stats["clip"].append(((ratio - 1).abs() > clip).float().mean())
            stats = {k_: [x.item() if torch.is_tensor(x) else x for x in v_] for k_, v_ in stats.items()}
        if distill and it == self.distill_until:
            self.teacher = None
        self.upd_s = time.time() - t1
        self.report(stats, n)
        if it % hp.get("league_every", 25) == 0:
            self.snaps.append(frozen(model))
            self.snaps[:] = self.snaps[:1] + self.snaps[1:][-(hp.get("league_size", 8) - 1):]
            self.save(self.out / "league" / f"it{it:05d}.pt")
            self.assign_league()
        if it % hp.get("save_every", 10) == 0:
            self.save(self.out / f"ckpt_{it:05d}.pt")
            self.save(self.out / "latest.pt")

    def report(self, stats, n):
        """Print the "it N ..." training line (other tools grep it) and write the TensorBoard scalars."""
        life = self.env.pop_stats(self.arenas)
        now = time.time()
        dt, self.t_last = now - self.t_last, now
        mean = lambda xs: statistics.fmean(xs) if xs else 0.0
        steps = self.T * self.skip * self.nb * self.env.P
        scal = {
            "play/life_return": life["ret"], "play/peak_mass": life["peak"], "play/life_steps": life["steps"],
            "play/death_share": life["died_share"], "play/virus_shots": life["shots"], "play/virus_pops": life["pops"],
            "play/kills": life["kills"], "play/stuck_share": life["stuck_share"], "play/peak_mass_max": life["peak_max"],
            "play/lives_over_10k": life["lives_over_10k"],
            "play/split_rate": float(self.splits) / max(n, 1), "play/eject_rate": float(self.ejects) / max(n, 1),
            "train/entropy": mean(stats["ent"]), "train/kl": mean(stats["kl"]), "train/clip_frac": mean(stats["clip"]),
            "train/value_loss": mean(stats["vf"]),
            "speed/samples_per_s": n / dt, "speed/agent_steps_per_s": steps / dt,
        }
        for key, val in scal.items():
            self.tb.add_scalar(key, val, self.it)
        self.tb.flush()
        self.print(f"it {self.it:5d} samples {n:6d} ({n / dt:6.0f}/s, {steps / dt:6.0f} agent-steps/s, collect "
                   f"{self.collect_s:4.1f}s upd {self.upd_s:4.1f}s) lives {life['lives']:5.0f} peak {life['peak']:6.0f} "
                   f"max {life['peak_max']:6.0f} >10k {life['lives_over_10k']:3.0f} len {life['steps']:5.0f} "
                   f"died {life['died_share']:.2f} kills {life['kills']:.2f} shots {life['shots']:.2f} pops {life['pops']:.2f} "
                   f"stuck {life['stuck_share']:.2f} split {scal['play/split_rate']:.3f} eject {scal['play/eject_rate']:.3f} "
                   f"ent {scal['train/entropy']:.2f} kl {scal['train/kl']:.4f} clip {scal['train/clip_frac']:.2f} "
                   f"vf {scal['train/value_loss']:.4f} maxlogr {max(stats['maxlogr'], default=0):.1f} vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G")


def collect(env: SimEnv, learners: list[Learner], obs, T: int, skip: int):
    """T decisions of every learner (and its league) plus the scripted bots, each held for `skip`
    env steps. Returns the observation after the last step."""
    g, s, v = obs
    A, dev = env.n_agents, env.dev
    bot_idx = torch.cat([L.bot_idx for L in learners]) if learners else None
    for L in learners:
        L.begin(T, obs_for(L.model, env, g).shape[1:])
    for t in range(T):
        acts = torch.zeros(A, 3, dtype=torch.long, device=dev)
        for L in learners:
            L.act(t, g, s, v, acts)
        if len(bot_idx):
            acts[bot_idx] = scripted_actions(env.w, env.bot_split_prob).reshape(-1, 3)[bot_idx]
        R = torch.zeros(A, device=dev)
        D = torch.zeros(A, dtype=torch.bool, device=dev)
        Tr = torch.zeros(A, dtype=torch.bool, device=dev)
        for j in range(skip):
            if j == 1:                       # split / eject fire once per decision
                acts = acts.clone()
                acts[:, 1:] = 0
            g2, s2, r, d, v2 = env.step(acts, observe=j == skip - 1)
            R += torch.where(D, torch.zeros_like(r), r)    # after a death: the next life's
            Tr |= env.trunc & ~D
            D |= d
        for L in learners:
            L.record(t, R, D & ~Tr, Tr)
        g, s, v = g2, s2, v2
    for L in learners:
        L.finish(g, s, v)
    return g, s, v


def setup_cuda(vram_gb: float, seed: int = 7):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2 ** 30 / total))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    """All command-line flags of `python -m rl.ppo_sim` (defaults are the single source of truth)."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="s1")
    p.add_argument("--init", help="checkpoint to start from (guided or sim-layout)")
    p.add_argument("--b", type=int, default=256, help="arenas")
    p.add_argument("--p", type=int, default=16, help="players per arena")
    p.add_argument("--steps", type=int, default=32, help="decisions per iteration")
    p.add_argument("--skip", type=int, default=1, help="env steps (80 ms) per decision")
    p.add_argument("--episode-steps", type=int, default=20000,
                   help="arena reset period in steps (2 ticks = 80 ms each); long, so lives can reach 10k+ mass")
    p.add_argument("--league-frac", type=float, default=0.2, help="arenas where half the slots play league snapshots")
    p.add_argument("--bot-frac", type=float, default=0.1, help="arenas where half the slots are scripted bots")
    for k in ("farm", "hunt", "flee", "giant"):
        p.add_argument(f"--{k}-frac", type=float, default=0.0, help=f"{k} drill arenas (see layout())")
    p.add_argument("--farm-players", type=int, default=LAYOUT_DEFAULTS["farm_players"])
    p.add_argument("--hunt-players", type=int, default=LAYOUT_DEFAULTS["hunt_players"])
    p.add_argument("--flee-players", type=int, default=LAYOUT_DEFAULTS["flee_players"])
    p.add_argument("--flee-hunters", type=int, default=LAYOUT_DEFAULTS["flee_hunters"])
    p.add_argument("--giants", type=int, default=LAYOUT_DEFAULTS["giants"])
    p.add_argument("--giant-learners", type=int, default=LAYOUT_DEFAULTS["giant_learners"])
    p.add_argument("--league-every", type=int, default=25)
    p.add_argument("--league-size", type=int, default=8, help="snapshots kept in memory")
    p.add_argument("--league-models", type=int, default=4, help="distinct snapshots playing at once")
    p.add_argument("--iters", type=int, default=10 ** 9)
    p.add_argument("--max-minutes", type=float, default=0, help="stop after this much wall time (0 = no limit)")
    p.add_argument("--start-it", type=int, default=0, help="resume numbering (with --init <run>/latest.pt)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--mb", type=int, default=2048, help="minibatch size (samples); 2048 in bf16 is ~2.6x faster than 1024 fp32")
    p.add_argument("--no-amp", action="store_true", help="float32 network passes")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.998, help="per env step (applied as gamma ** skip per decision)")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--vf-warmup", type=int, default=10, help="iterations training only the critic")
    p.add_argument("--explore", type=float, default=0.02, help="split/eject exploration floor (Policy.binary_eps)")
    p.add_argument("--virus-reward", default="0,0,0", help="shot,pop,popmass weights")
    p.add_argument("--explore-reward", type=float, default=0.0, help="bonus per new 500x500 square in a life")
    p.add_argument("--curriculum-p", type=float, default=0.5, help="share of respawns that start big")
    p.add_argument("--curriculum-mass", default="50,5000", help="their start mass range (log-uniform)")
    p.add_argument("--arena-steps", type=int, default=0,
                   help="each arena restarts on its own (staggered) clock after this many steps (0: off)")
    p.add_argument("--farm-spawn", help="farm arenas: (re)start mass range, e.g. 10,3000")
    p.add_argument("--split-explore-min-mass", type=float, default=70.0, help="split exploration floor from this mass")
    p.add_argument("--no-eject-explore", action="store_true", help="no eject exploration floor")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--arch", help="architecture genes as JSON (rl.model.Policy kwargs: width, convs, pool, "
                   "in_mask, s_mask, shared); differs from --init's -> fresh net distilled from it")
    p.add_argument("--distill-iters", type=int, default=30, help="iterations of imitating --init when --arch differs")
    p.add_argument("--hof", help="directory of checkpoints that always stay in the league (e.g. past evolution winners)")
    p.add_argument("--growth", help="growth reward instead of linear mass: 'par_doubling_min,par_mass,death' "
                   "e.g. 4,500,1 = log-mass change, minus par (doubling every 4 min) above 500 mass, death costs "
                   "its log-mass gain + 1")
    p.add_argument("--action-bonus", default="0,0", help="reward per split, per eject (W) the game carries out")
    p.add_argument("--kill-reward", default="0,0", help="per enemy cell eaten, per 200 enemy mass eaten")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--bc-farmer-iters", type=int, default=0,
                   help="first imitate the nearest-pellet farmer (sim.bots.farmer_actions; no splits) for this "
                        "many iterations: DAgger on the net's own states")
    p.add_argument("--bc-lr", type=float, default=1e-3, help="learning rate while imitating")
    p.add_argument("--bc-fade-iters", type=int, default=0,
                   help="after imitating: the imitation term fades out linearly over this many PPO iterations")
    p.add_argument("--farm-eval-every", type=int, default=0,
                   help="every N iterations: the farming-alone check (sim.tourney.farm), printed and in TensorBoard")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vram-gb", type=float, default=12.0,
                   help="hard VRAM cap: past it torch raises OOM instead of Windows spilling into system RAM")
    return p


def build_start_model(args, dev: str) -> tuple[Policy, Policy | None]:
    """The network to train and, when `--arch` differs from `--init`'s, the frozen parent to distil from."""
    teacher = None
    if args.arch:
        model = new_model(**json.loads(args.arch)).to(dev)
        if args.init:
            parent = from_checkpoint(args.init, dev)
            if parent.config == model.config:
                model = parent
            else:
                teacher = frozen(parent)
                print(f"new architecture {model.config} - distilling from {args.init} for "
                      f"{args.distill_iters} iterations", flush=True)
    else:
        model = from_checkpoint(args.init, dev) if args.init else new_model(args.width).to(dev)
    return model, teacher


def build_hparams(args) -> dict:
    """Learner hyper-parameters: the flags `Learner` reads, with comma-separated ones parsed to lists."""
    floats = lambda x: [float(v) for v in x.split(",")]
    hp = {k: getattr(args, k) for k in ("lr", "ent", "gamma", "lam", "clip", "epochs", "mb", "target_kl", "vf",
                                         "vf_warmup", "explore", "league_every", "league_size", "league_models",
                                         "save_every", "explore_reward", "bc_farmer_iters", "bc_lr", "bc_fade_iters",
                                         *LAYOUT_DEFAULTS)}
    hp.update(virus_reward=floats(args.virus_reward), kill_reward=floats(args.kill_reward),
              action_bonus=floats(args.action_bonus), growth=args.growth)
    hp.update(farm_spawn=floats(args.farm_spawn) if args.farm_spawn else None,
              split_explore_min_mass=args.split_explore_min_mass, eject_explore=not args.no_eject_explore)
    return hp


def farm_check(L: "Learner", out: Path) -> None:
    """The farming-alone check (`--farm-eval-every`): save, play the net alone from 10 and from 1000 mass."""
    from sim.tourney import farm
    L.save(out / "latest.pt")
    net = {"net": str(out / "latest.pt")}
    fm = farm(net, B=64, free=True)["net"]
    fb = farm(net, B=64, free=True, start=1000.0, gain=True)["net"]
    L.tb.add_scalar("eval/farm_mass", fm, L.it)
    L.tb.add_scalar("eval/farm_gain_from_1000", fb, L.it)
    print(f"farm check at it {L.it}: from 10 -> {fm:.0f} mass in 2 min (routines 283-310); "
          f"from 1000: +{fb:.0f} in 2 min (routine +374, split-farming routine +467)", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    global AMP
    AMP = not args.no_amp
    setup_cuda(args.vram_gb, args.seed)
    dev = "cuda"
    out = ROOT / "runs" / "rl" / args.run
    model, teacher = build_start_model(args, dev)
    hp = build_hparams(args)
    B, P, T = args.b, args.p, args.steps
    hof = []
    if args.hof:
        hof = [frozen(from_checkpoint(str(f), dev)) for f in sorted(Path(args.hof).glob("*.pt"))]
        print(f"hall of fame: {len(hof)} opponents from {args.hof}", flush=True)
    kinds = {obs_kind(m) for m in [model, *hof]}
    env = SimEnv(B=B, P=P, seed=args.seed, episode_steps=args.episode_steps, curriculum_p=args.curriculum_p,
                 curriculum_mass=tuple(float(x) for x in args.curriculum_mass.split(",")),
                 obs_dtype=torch.float16, skip=args.skip, obs="both" if len(kinds) > 1 else kinds.pop(),
                 arena_steps=args.arena_steps)
    L = Learner(args.run, model, env, 0, B, hp, out, hof=hof, teacher=teacher,
                distill_iters=args.distill_iters if teacher is not None else 0, start_it=args.start_it,
                skip=args.skip)
    g, s, _, _, v = env.reset()
    print(f"PPO-sim run {args.run}: {B} arenas x {P} slots ({L.N} learners, {len(L.bot_idx)} bots, "
          f"{sum(len(i) for i in L.league_idx)} league), {T} decisions/iter, skip {args.skip}, "
          f"episode {args.episode_steps} steps", flush=True)
    t_start = time.time()
    for _ in range(args.iters):
        if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
            break
        g, s, v = collect(env, [L], (g, s, v), T, args.skip)
        L.update()
        if args.farm_eval_every and L.it % args.farm_eval_every == 0:
            farm_check(L, out)
    L.save(out / "latest.pt")
    print(f"done at iteration {L.it}", flush=True)


if __name__ == "__main__":
    main()
