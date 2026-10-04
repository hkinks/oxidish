"""PPO self-play fully on the GPU, on the batched simulator (sim/).

    uv run python -m rl.ppo_sim --run s1 [--init runs/rl/g8_048.pt] [--b 256] [--steps 32]

B arenas x P players step on the GPU (`sim.env.SimEnv`); observations, rollouts and PPO all stay
on the GPU. Slot roles: in most arenas every slot is a learner playing the current policy
(self-play). A fraction of arenas gives half of the slots to opponents instead: `--league-frac`
arenas use frozen snapshots of this run (a league: the starting policy plus a snapshot every
`--league-every` iterations), `--bot-frac` arenas use the scripted bot (sim.bots). Only learner
slots produce training samples. Lives' statistics (peak mass, shots, pops, ...) are logged for the
learner slots only.

Network: rl.model.Policy with the sim layout (c=15: the guided net's 13 channels + killable prey +
enemy mass density; s=16, g=48) and heads (17, 2, 2): 16
headings + stay, split, eject. There is no FOLLOW action (a net trained here has no champion), so the
move head has 17 outputs instead of the guided net's 18 and FOLLOW can never be chosen. The 8 hint
scalars are zero in the sim and in rl.nohint.NoHintEncoder (the real-server side). `--init` takes a
guided checkpoint (its FOLLOW logit row is dropped, its hint input weights stay but see zeros) or an
earlier run of this script.

Checkpoints: runs/rl/<run>/ckpt_NNNNN.pt and latest.pt, written with rl.common.save and tagged
`sim=True, hint=False`; rl.common.load reads them and rl.policy.NNPolicy plays them on the real
server (c=13 and no follow -> rl.nohint.NoHintEncoder).
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sim.bots import scripted_actions
from sim.env import SimEnv

from .common import ROOT, load, save
from .model import Policy

C_IN, S_IN, G_IN = 15, 16, 48
ARCH_KEYS = ("width", "convs", "pool", "in_mask", "s_mask", "shared")
HEADS = (17, 2, 2)


def new_model(width: int = 256, **arch) -> Policy:
    return Policy(width=width, c=C_IN, s=S_IN, heads=HEADS, follow=None, g=G_IN, **arch)


def arch_of(model: Policy) -> dict:
    return {k: model.config[k] for k in ARCH_KEYS if k in model.config}


def from_checkpoint(path: str, dev: str) -> Policy:
    """A sim-layout policy initialised from a guided (18 moves, FOLLOW) or sim checkpoint.
    Nets from before the extra input channels (c=13) get zero weights for them, so they play
    exactly as before and can learn to use them."""
    src = load(path, dev)
    cfg = src.config
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
    return model


AMP = True   # bfloat16 network passes (see Policy.dists); --no-amp turns it off


def act_sampled(model: Policy, g: torch.Tensor, s: torch.Tensor):
    with torch.no_grad():
        return model.act(g if AMP else g.float(), s, amp=AMP)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="s1")
    p.add_argument("--init", help="checkpoint to start from (guided or sim-layout)")
    p.add_argument("--b", type=int, default=256, help="arenas")
    p.add_argument("--p", type=int, default=16, help="players per arena")
    p.add_argument("--steps", type=int, default=32, help="env steps per iteration")
    p.add_argument("--episode-steps", type=int, default=20000,
                   help="arena reset period in steps (2 ticks = 80 ms each); long, so lives can reach 10k+ mass")
    p.add_argument("--league-frac", type=float, default=0.2, help="arenas where half the slots play league snapshots")
    p.add_argument("--farm-frac", type=float, default=0.0,
                   help="arenas with only --farm-players learners and nothing else: a clean farming signal")
    p.add_argument("--farm-players", type=int, default=3)
    p.add_argument("--hunt-frac", type=float, default=0.0,
                   help="hunting drills: arenas where --hunt-players learners start big (400-2000 mass) among "
                        "small scripted bots (20-200): splitting onto prey pays off right away")
    p.add_argument("--hunt-players", type=int, default=4)
    p.add_argument("--bot-frac", type=float, default=0.1, help="arenas where half the slots are scripted bots")
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
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--vf-warmup", type=int, default=10, help="iterations training only the critic")
    p.add_argument("--explore", type=float, default=0.02, help="split/eject exploration floor (Policy.binary_eps)")
    p.add_argument("--virus-reward", default="0,0,0", help="shot,pop,popmass weights")
    p.add_argument("--explore-reward", type=float, default=0.0, help="bonus per new 500x500 square in a life")
    p.add_argument("--curriculum-p", type=float, default=0.5, help="share of respawns that start big")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--arch", help="architecture genes as JSON (rl.model.Policy kwargs: width, convs, pool, "
                   "in_mask, s_mask, shared); differs from --init's -> fresh net distilled from it")
    p.add_argument("--distill-iters", type=int, default=30, help="iterations of imitating --init when --arch differs")
    p.add_argument("--hof", help="directory of checkpoints that always stay in the league (e.g. past evolution winners)")
    p.add_argument("--kill-reward", default="0,0", help="per enemy cell eaten, per 200 enemy mass eaten")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vram-gb", type=float, default=12.0,
                   help="hard VRAM cap: past it torch raises OOM instead of Windows spilling into system RAM")
    args = p.parse_args()
    global AMP
    AMP = not args.no_amp

    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2 ** 30 / total))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    out = ROOT / "runs" / "rl" / args.run
    teacher = None
    if args.arch:
        arch = json.loads(args.arch)
        model = new_model(**arch).to(dev)
        if args.init:
            parent = from_checkpoint(args.init, dev)
            if parent.config == model.config:
                model = parent
            else:   # new architecture: imitate the parent first (DAgger: our own states, its actions)
                teacher = parent.eval()
                for q in teacher.parameters():
                    q.requires_grad_(False)
                print(f"new architecture {model.config} - distilling from {args.init} for "
                      f"{args.distill_iters} iterations", flush=True)
    else:
        model = from_checkpoint(args.init, dev) if args.init else new_model(args.width).to(dev)
    model.binary_eps = args.explore
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    B, P, T = args.b, args.p, args.steps
    env = SimEnv(B=B, P=P, seed=args.seed, episode_steps=args.episode_steps, curriculum_p=args.curriculum_p,
                 obs_dtype=torch.float16, virus_reward=tuple(float(x) for x in args.virus_reward.split(",")),
                 explore_reward=args.explore_reward,
                 kill_reward=tuple(float(x) for x in args.kill_reward.split(",")))
    # ---- slot roles: 0 learner (current policy), 1 league snapshot, 2 scripted bot
    n_league, n_bot = round(B * args.league_frac), round(B * args.bot_frac)
    n_farm, n_hunt = round(B * args.farm_frac), round(B * args.hunt_frac)
    n_league = max(0, min(n_league, B - n_farm - n_hunt - n_bot))   # drills come first
    role = torch.zeros(B, P, dtype=torch.long, device=dev)
    half = P - P // 2
    role[B - n_league - n_bot:B - n_bot, half:] = 1
    role[B - n_bot:, half:] = 2
    if n_farm:                                      # farm arenas: the first n_farm (all-learner) arenas
        env.active[:n_farm, args.farm_players:] = False
        role[:n_farm, args.farm_players:] = 3          # inactive: no actions, no samples
    if n_hunt:                                      # hunt arenas: right after the farm arenas
        h = slice(n_farm, n_farm + n_hunt)
        role[h, args.hunt_players:] = 2
        env.spawn_mass[h, :args.hunt_players] = torch.tensor([400.0, 2000.0], device=dev)
        env.spawn_mass[h, args.hunt_players:] = torch.tensor([20.0, 200.0], device=dev)
    assert n_farm + n_hunt + n_league + n_bot <= B, "drill + league + bot arenas exceed --b"
    flat = role.reshape(-1)
    learn_idx = (flat == 0).nonzero()[:, 0]
    league_arenas = list(range(B - n_league - n_bot, B - n_bot))
    bot_idx = (flat == 2).nonzero()[:, 0]
    N = len(learn_idx)
    env.set_tracked((role == 0))
    def frozen(m: Policy) -> Policy:
        m = copy.deepcopy(m).eval()
        m.binary_eps = 0.0
        for q in m.parameters():
            q.requires_grad_(False)
        return m

    snaps = [frozen(model)]                                 # league: starting policy + periodic snapshots
    old = sorted((out / "league").glob("it*.pt"))          # resuming: the run's earlier snapshots rejoin
    if old:
        keep = old[:1] + old[1:][-(args.league_size - 1):]
        snaps = [frozen(from_checkpoint(str(f), dev)) for f in keep]
        print(f"league resumed from {len(keep)} snapshots: {', '.join(f.stem for f in keep)}", flush=True)
    hof = []                                                # hall of fame: fixed opponents (other runs' bests)
    if args.hof:
        for f in sorted(Path(args.hof).glob("*.pt")):
            hof.append(frozen(from_checkpoint(str(f), dev)))
        print(f"hall of fame: {len(hof)} opponents from {args.hof}", flush=True)
    league_models, league_idx = [], []
    if n_league:
        k = min(args.league_models, n_league)
        chunks = np.array_split(np.array(league_arenas), k)
        for ch in chunks:
            ix = (role[torch.as_tensor(ch, device=dev)] == 1).nonzero()
            league_idx.append(torch.as_tensor(ch, device=dev)[ix[:, 0]] * P + ix[:, 1])
            league_models.append(None)

    def assign_league():
        pool = snaps + hof
        for i in range(len(league_models)):
            league_models[i] = pool[np.random.randint(len(pool))]
    assign_league()

    tb = SummaryWriter(str(ROOT / "logs" / "tb" / args.run))
    g, s, _, _, v = env.reset()
    A = B * P
    print(f"PPO-sim run {args.run}: {B} arenas x {P} slots = {A} agents ({N} learners, {len(bot_idx)} bots, "
          f"{sum(len(i) for i in league_idx)} league), {T} steps/iter, episode {args.episode_steps} steps",
          flush=True)
    if not (out / "league" / "it00000.pt").exists():
        save(model, out / "league" / "it00000.pt", it=0, sim=True, hint=False)

    gdim = g.shape[1:]
    t_start = time.time()
    it = args.start_it
    for it in range(args.start_it + 1, args.start_it + args.iters + 1):
        if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
            it -= 1
            break
        t0 = time.time()
        G = torch.empty((T, N, *gdim), dtype=torch.float16, device=dev)
        Sc = torch.empty((T, N, S_IN), device=dev)
        Act = torch.zeros((T, N, 3), dtype=torch.long, device=dev)
        Logp = torch.zeros((T, N), device=dev)
        Val = torch.zeros((T + 1, N), device=dev)
        Rew = torch.zeros((T, N), device=dev)
        Done = torch.zeros((T, N), device=dev)
        Valid = torch.zeros((T + 1, N), device=dev)
        splits = ejects = torch.zeros((), device=dev)
        for t in range(T):
            gl, sl = g[learn_idx], s[learn_idx]
            a, logp, val = act_sampled(model, gl, sl)
            G[t], Sc[t], Act[t], Logp[t], Val[t] = gl, sl, a, logp, val
            vl = v[learn_idx]
            Valid[t] = vl.float()
            splits = splits + (a[:, 1] * vl).sum()
            ejects = ejects + (a[:, 2] * vl).sum()
            acts = torch.zeros(A, 3, dtype=torch.long, device=dev)
            acts[learn_idx] = a
            for m, ix in zip(league_models, league_idx):
                acts[ix] = act_sampled(m, g[ix], s[ix])[0]
            if len(bot_idx):
                acts[bot_idx] = scripted_actions(env.w, env.bot_split_prob).reshape(-1, 3)[bot_idx]
            g, s, r, d, v = env.step(acts)
            Rew[t] = r[learn_idx]
            Done[t] = d[learn_idx].float()
        with torch.no_grad():
            Val[T] = act_sampled(model, g[learn_idx], s[learn_idx])[2]
        Valid[T] = v[learn_idx].float()
        life = env.pop_stats()
        collect_s = time.time() - t0

        adv = torch.zeros((T, N), device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            cont = (1 - Done[t]) * Valid[t + 1]
            delta = Rew[t] + args.gamma * Val[t + 1] * cont - Val[t]
            last = delta + args.gamma * args.lam * cont * last
            adv[t] = last * Valid[t]
        ret = adv + Val[:T]

        ids = (Valid[:T].reshape(-1) > 0).nonzero()[:, 0]          # indices of valid samples
        n = len(ids)
        Gf, Sf, Af = G.view(T * N, *gdim), Sc.view(T * N, -1), Act.view(T * N, 3)
        Lf, advf, retf = Logp.view(-1), adv.view(-1), ret.view(-1)
        a_ = advf[ids]
        advn = torch.zeros_like(advf)
        advn[ids] = (a_ - a_.mean()) / (a_.std() + 1e-8)
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": []}
        nmb = max(n // args.mb, 1)
        for ep in range(args.epochs):
            if stats["kl"] and statistics.fmean(stats["kl"][-nmb:]) > args.target_kl:
                break
            perm = ids[torch.randperm(n, device=dev)]
            for k in range(0, n, args.mb):
                ix = perm[k:k + args.mb]
                dists, value = model.dists(Gf[ix] if AMP else Gf[ix].float(), Sf[ix], amp=AMP)
                logp, ent = model.logp_entropy(dists, Af[ix])
                ent = ent.mean()
                ratio = (logp - Lf[ix]).exp()
                pg = -torch.min(ratio * advn[ix], ratio.clamp(1 - args.clip, 1 + args.clip) * advn[ix]).mean()
                vf = 0.5 * (value - retf[ix]).pow(2).mean()
                if teacher is not None and it <= args.start_it + args.distill_iters:
                    with torch.no_grad():
                        tdists, _ = teacher.dists(Gf[ix] if AMP else Gf[ix].float(), Sf[ix], amp=AMP)
                    kl_t = sum(torch.distributions.kl_divergence(td, sd_).mean() for td, sd_ in zip(tdists, dists))
                    loss = kl_t + args.vf * vf
                elif it <= args.vf_warmup:
                    loss = args.vf * vf
                else:
                    loss = pg + args.vf * vf - args.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.detach()); stats["vf"].append(vf.detach()); stats["ent"].append(ent.detach())
                    stats["kl"].append(((ratio - 1) - ratio.log()).mean())
                    stats["clip"].append(((ratio - 1).abs() > args.clip).float().mean())
            stats = {k_: [x.item() if torch.is_tensor(x) else x for x in v_] for k_, v_ in stats.items()}  # one sync per epoch
        stats = {k_: [x.item() if torch.is_tensor(x) else x for x in v_] for k_, v_ in stats.items()}
        dt = time.time() - t0
        mean = lambda xs: statistics.fmean(xs) if xs else 0.0
        valid_n = n
        scal = {
            "play/life_return": life["ret"], "play/peak_mass": life["peak"], "play/life_steps": life["steps"],
            "play/death_share": life["died_share"], "play/virus_shots": life["shots"], "play/virus_pops": life["pops"], "play/kills": life["kills"],
            "play/stuck_share": life["stuck_share"], "play/peak_mass_max": life["peak_max"],
            "play/lives_over_10k": life["lives_over_10k"],
            "play/split_rate": float(splits) / max(valid_n, 1), "play/eject_rate": float(ejects) / max(valid_n, 1),
            "play/follow_rate": 0.0,
            "train/entropy": mean(stats["ent"]), "train/kl": mean(stats["kl"]), "train/clip_frac": mean(stats["clip"]),
            "train/value_loss": mean(stats["vf"]), "train/anchor_kl": 0.0,
            "speed/samples_per_s": valid_n / dt, "speed/agent_steps_per_s": A * T / dt,
        }
        for key, val in scal.items():
            tb.add_scalar(key, val, it)
        tb.flush()
        print(f"it {it:5d} samples {valid_n:6d} ({valid_n / dt:6.0f}/s, {A * T / dt:6.0f} agent-steps/s, collect "
              f"{collect_s:4.1f}s upd {dt - collect_s:4.1f}s) lives {life['lives']:5.0f} peak {life['peak']:6.0f} "
              f"max {life['peak_max']:6.0f} >10k {life['lives_over_10k']:3.0f} len {life['steps']:5.0f} "
              f"died {life['died_share']:.2f} kills {life['kills']:.2f} shots {life['shots']:.2f} pops {life['pops']:.2f} "
              f"stuck {life['stuck_share']:.2f} split {scal['play/split_rate']:.3f} eject {scal['play/eject_rate']:.3f} "
              f"ent {scal['train/entropy']:.2f} kl {scal['train/kl']:.4f} clip {scal['train/clip_frac']:.2f} "
              f"vf {scal['train/value_loss']:.4f} vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if it % args.league_every == 0:
            snaps.append(frozen(model))
            snaps[:] = snaps[:1] + snaps[1:][-(args.league_size - 1):]
            save(model, out / "league" / f"it{it:05d}.pt", it=it, sim=True, hint=False)
            assign_league()
        if it % args.save_every == 0:
            save(model, out / f"ckpt_{it:05d}.pt", it=it, sim=True, hint=False)
            save(model, out / "latest.pt", it=it, sim=True, hint=False)

    save(model, out / "latest.pt", it=it, sim=True, hint=False)
    print(f"done at iteration {it}", flush=True)


if __name__ == "__main__":
    main()
