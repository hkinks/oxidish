"""PPO self-play training on the arena.

    uv run python -m rl.ppo --run p1 --init runs/rl/bc.pt

Learner slots in every arena share the current policy (self-play); opponents
are evolved champions; the server adds its built-in bots. Each iteration
collects `--steps` env steps from all slots, then runs PPO epochs on the GPU.
Checkpoints go to runs/rl/<run>/; `--agents nn=runs/rl/<run>/latest.pt:N`
plays them in arena tournaments.
"""

import argparse
import copy
import json
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .common import ROOT, env_cfgs, load, save
from .model import Policy
from .vec import VecEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="p1")
    p.add_argument("--init", help="checkpoint to start from (e.g. runs/rl/bc.pt)")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--n-learn", type=int, default=4)
    p.add_argument("--n-opp", type=int, default=3)
    p.add_argument("--episode-ticks", type=int, default=9000, help="arena reset period (6 game-min)")
    p.add_argument("--steps", type=int, default=256, help="env steps per iteration")
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-kl", type=float, default=0.02, help="stop epochs early past this KL")
    p.add_argument("--gamma", type=float, default=0.998, help="~40 s horizon at 80 ms/step")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--vf-warmup", type=int, default=10, help="iterations training only the critic")
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--explore", type=float, default=0.02,
                   help="exploration floor for split/eject (each gets >= explore/2)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--anchor", help="frozen reference policy to stay close to: a checkpoint, or 'init'")
    p.add_argument("--strong", action="store_true",
                   help="opponents are only the guide champion and its hunter")
    p.add_argument("--opp-every", type=int, default=2, help="opponents decide every n-th step")
    p.add_argument("--virus-reward", default="0,0,0",
                   help="shot,pop,popmass: bonus per virus we shot, per enemy exploded on it, "
                        "and per popped mass (x mass reward)")
    p.add_argument("--explore-reward", type=float, default=0.0,
                   help="bonus per newly visited 500x500 map square in a life (novelty)")
    p.add_argument("--league", action="store_true",
                   help="half the opponents are frozen snapshots of this run (self-play league)")
    p.add_argument("--league-every", type=int, default=15)
    p.add_argument("--follow-cost", type=float, default=0.0,
                   help="reward penalty per FOLLOW step (guided): pushes towards own moves")
    p.add_argument("--guide", help="champion json for a guided (residual) net")
    p.add_argument("--follow-bias", type=float, default=7.0, help="initial FOLLOW logit (guided)")
    p.add_argument("--anchor-kl", type=float, default=0.5,
                   help="weight of KL(anchor || policy); PPO moves away only where advantages insist")
    args = p.parse_args()

    dev = "cuda"
    out = ROOT / "runs" / "rl" / args.run
    guide = json.loads(Path(args.guide).read_text()) if args.guide else None
    if args.init:
        model = load(args.init, dev)
        guide = guide or model.guide
    elif guide:  # fresh guided net that starts out following the champion
        from .guided import C as GC, FOLLOW, G as GG, HEADS as GH, S as GS
        model = Policy(c=GC, s=GS, heads=GH, follow=FOLLOW, g=GG).to(dev)
        model.init_follow(args.follow_bias)
    else:
        model = Policy().to(dev)
    model.binary_eps = args.explore
    anchor = None
    if args.anchor == "init":  # stay close to the starting policy
        anchor = copy.deepcopy(model).eval()
    elif args.anchor:
        anchor = load(args.anchor, dev).eval()
    if anchor is not None:
        for prm in anchor.parameters():
            prm.requires_grad_(False)
    league = out / "league" if args.league else None
    if league:  # league seeded with the starting policy; snapshots join every --league-every
        save(model, league / "it00000.pt", it=0, guide=guide)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    env = VecEnv(env_cfgs(args.workers, args.n_learn, args.n_opp, args.episode_ticks, args.seed,
                          guide=guide, strong=args.strong, opp_every=args.opp_every,
                          league=str(league) if league else None,
                          virus_reward=tuple(float(x) for x in args.virus_reward.split(",")),
                          explore_reward=args.explore_reward))
    g, s, _, _, v, _, _ = env.reset()
    N, T = len(v), args.steps
    lives = deque(maxlen=200)
    tb = SummaryWriter(str(ROOT / "logs" / "tb" / args.run))  # tensorboard --logdir logs/tb
    print(f"PPO run {args.run}: {args.workers} arenas x {args.n_learn} learners = {N} agents, "
          f"{T} steps/iter", flush=True)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        G = torch.zeros((T, N, *g.shape[1:]), dtype=torch.float16, device=dev)
        Sc = torch.zeros((T, N, s.shape[1]), device=dev)
        Act = torch.zeros((T, N, 3), dtype=torch.long, device=dev)
        Logp = torch.zeros((T, N), device=dev)
        Val = torch.zeros((T + 1, N), device=dev)
        Rew = torch.zeros((T, N), device=dev)
        Done = torch.zeros((T, N), device=dev)
        Valid = torch.zeros((T + 1, N), device=dev)
        n_split = n_eject = 0
        for t in range(T):
            gt = torch.from_numpy(g).to(dev); st = torch.from_numpy(s).to(dev)
            a, logp, val = model.act(gt, st)
            G[t], Sc[t], Act[t], Logp[t], Val[t] = gt.half(), st, a, logp, val
            Valid[t] = torch.from_numpy(v).to(dev)
            acts = a.cpu().numpy()
            n_split += int((acts[:, 1] * v).sum()); n_eject += int((acts[:, 2] * v).sum())
            g, s, r, d, v, _, fin = env.step(acts)
            Rew[t] = torch.from_numpy(r).to(dev)
            if args.follow_cost and model.follow is not None:
                # Price on copying the champion, so the net has to find its own better moves.
                Rew[t] -= args.follow_cost * (a[:, 0] == model.follow).float() * Valid[t]
            Done[t] = torch.from_numpy(d.astype(np.float32)).to(dev)
            lives.extend(fin)
        with torch.no_grad():
            _, _, Val[T] = model.act(torch.from_numpy(g).to(dev), torch.from_numpy(s).to(dev))
        Valid[T] = torch.from_numpy(v).to(dev)
        collect_s = time.time() - t0

        # GAE. Transition t: obs_t -> reward/done observed at t+1. A done (death or arena
        # reset) or an invalid next obs stops bootstrapping; invalid obs_t is masked out.
        adv = torch.zeros((T, N), device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            cont = (1 - Done[t]) * Valid[t + 1]
            delta = Rew[t] + args.gamma * Val[t + 1] * cont - Val[t]
            last = delta + args.gamma * args.lam * cont * last
            adv[t] = last * Valid[t]
        ret = adv + Val[:T]

        mask = Valid[:T].reshape(-1) > 0
        bG, bS = G.reshape(T * N, *G.shape[2:])[mask], Sc.reshape(T * N, -1)[mask]
        bA, bL = Act.reshape(T * N, 3)[mask], Logp.reshape(-1)[mask]
        bAdv, bRet = adv.reshape(-1)[mask], ret.reshape(-1)[mask]
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        n = len(bA)
        mb = max(n // args.minibatches, 1)
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "akl": []}
        for _ in range(args.epochs):
            if stats["kl"] and statistics.fmean(stats["kl"][-args.minibatches:]) > args.target_kl:
                break  # policy moved enough this iteration
            perm = torch.randperm(n, device=dev)
            for k in range(0, n, mb):
                ix = perm[k:k + mb]
                dists, value = model.dists(bG[ix].float(), bS[ix])
                logp, ent = model.logp_entropy(dists, bA[ix])
                ent = ent.mean()
                ratio = (logp - bL[ix]).exp()
                pg = -torch.min(ratio * bAdv[ix],
                                ratio.clamp(1 - args.clip, 1 + args.clip) * bAdv[ix]).mean()
                vf = 0.5 * (value - bRet[ix]).pow(2).mean()
                akl = torch.zeros((), device=dev)
                if anchor is not None:
                    with torch.no_grad():
                        ref, _ = anchor(bG[ix].float(), bS[ix])
                    for d, r in zip(dists, ref):
                        rp = r.log_softmax(-1)
                        akl = akl + (rp.exp() * (rp - d.logits.log_softmax(-1))).sum(-1).mean()
                loss = (args.vf * vf if it <= args.vf_warmup
                        else pg + args.vf * vf - args.ent * ent + args.anchor_kl * akl)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item())
                    stats["ent"].append(ent.item())
                    stats["kl"].append(((ratio - 1) - ratio.log()).mean().item())
                    stats["clip"].append(((ratio - 1).abs() > args.clip).float().mean().item())
                    stats["akl"].append(akl.item())

        valid_n = int(mask.sum())
        mean = lambda xs: statistics.fmean(xs) if xs else 0.0
        follow_rate = (float((bA[:, 0] == model.follow).float().mean())
                       if model.follow is not None else 0.0)
        for key, val in {
            "play/life_return": mean([l["ret"] for l in lives]),
            "play/peak_mass": mean([l["peak"] for l in lives]),
            "play/life_steps": mean([l["steps"] for l in lives]),
            "play/death_share": mean([float(l["died"]) for l in lives]),
            "play/split_rate": n_split / max(valid_n, 1),
            "play/eject_rate": n_eject / max(valid_n, 1),
            "play/follow_rate": follow_rate,
            "play/virus_shots": mean([l.get("shots", 0) for l in lives]),
            "play/virus_pops": mean([l.get("pops", 0) for l in lives]),
            "play/stuck_share": mean([l.get("stuck", 0) / max(l["steps"], 1) for l in lives]),
            "train/entropy": mean(stats["ent"]), "train/kl": mean(stats["kl"]),
            "train/clip_frac": mean(stats["clip"]), "train/value_loss": mean(stats["vf"]),
            "train/anchor_kl": mean(stats["akl"]),
            "speed/samples_per_s": valid_n / (time.time() - t0),
        }.items():
            tb.add_scalar(key, val, it)
        tb.flush()
        died = [l for l in lives if l["died"]]
        print(f"it {it:5d} samples {valid_n:5d} ({valid_n / (time.time() - t0):5.0f}/s, collect "
              f"{collect_s:4.1f}s) life ret {statistics.fmean(l['ret'] for l in lives) if lives else 0:6.2f} "
              f"peak {statistics.fmean(l['peak'] for l in lives) if lives else 0:6.0f} "
              f"len {statistics.fmean(l['steps'] for l in lives) if lives else 0:5.0f} "
              f"deaths {len(died)}/{len(lives)}  split {n_split / max(valid_n, 1):.3f} "
              f"eject {n_eject / max(valid_n, 1):.3f}  ent {statistics.fmean(stats['ent']):.2f} "
              f"kl {statistics.fmean(stats['kl']):.4f} clip {statistics.fmean(stats['clip']):.2f} "
              f"vf {statistics.fmean(stats['vf']):.4f} anchor-kl {statistics.fmean(stats['akl']):.3f}"
              f" follow {follow_rate:.3f} vshots {mean([l.get('shots', 0) for l in lives]):.2f}"
              f" vpops {mean([l.get('pops', 0) for l in lives]):.2f}"
              f" stuck {mean([l.get('stuck', 0) / max(l['steps'], 1) for l in lives]):.2f}",
              flush=True)
        if league and it % args.league_every == 0:
            save(model, league / f"it{it:05d}.pt", it=it, guide=guide)
        if it % args.save_every == 0:
            save(model, out / f"ckpt_{it:05d}.pt", it=it, guide=guide)
            save(model, out / "latest.pt", it=it, guide=guide)
    env.close()


if __name__ == "__main__":
    main()
