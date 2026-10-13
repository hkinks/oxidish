"""M4 phase B: PPO fine-tuning of a BC net on the ogar_py VecEnv (endless field), with a decaying teacher anchor.

    python rl/vision/train_ppo.py --name p1 --init runs/vision/a1/final.pt --hours 2

Policy: categorical over the 64x64 heat-map cells (logits = heat / tau); the executed target is the cell centre.
Split / eject are not used (always 0). Reward per decision: d log(mass) (clipped) + leader_bonus * [#1 in the world]
- death_penalty on a death. Value head pretrain first (trunk frozen, policy fixed), then PPO. Anchor: cross-entropy
of softmax(heat) against the planner's Gaussian heat target on the visited states, coefficient decaying linearly.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import FIELD, ROOT, TEACHER, pin_cores, target_metrics
from evalval import load_model
from model import VisionNet, decode_target, heat_target

import ogar_py

DT = 0.08

# League pool: the endless field (champ, r2g9, safe, ramp_r2, ramp300) + c3_4, shoot, two hunters and other rule-bot
# variants. DISJOINT from the held-out field (plan_c34, plan2_c34, vision_bc3, hunter_virus, see endlessH.sh).
# Each world draws `--pick` of them. (Vision-net opponents are not available inside ogar_py.)
LEAGUE = FIELD + [
    ("c3_4", "bot/c3_4.json", 1),
    ("shoot", "runs/ab/c34_shootthr.json", 1),
    ("hhunt", "bot/sparring/hunter_hunt.json", 1),
    ("hreck", "bot/sparring/hunter_reckless.json", 1),
    ("cm500", "runs/ab/c34_cm500.json", 1),
    ("esc", "runs/ab/c34_esc.json", 1),
    ("ev1", "runs/ab/c34_ev1.json", 1),
    ("fs", "runs/ab/c34_fs.json", 1),
    ("pd", "runs/ab/c3_4_pd.json", 1),
    ("wd400", "runs/ab/r300_wd400_ww800.json", 1),
    ("ww400pd", "runs/ab/ww400_pd.json", 1),
]


def cell_to_uv(idx, H=64, px=256):
    cell = px / H
    return torch.stack([((idx % H).float() + 0.5) * cell, ((idx // H).float() + 0.5) * cell], 1)


class Roll:
    def __init__(self, T, N, dev="cuda"):
        self.view = torch.zeros((T, N, 2, 256, 256), dtype=torch.uint8, device=dev)
        self.wide = torch.zeros((T, N, 2, 128, 128), dtype=torch.uint8, device=dev)
        self.teacher = torch.zeros((T, N, 4), device=dev)
        self.act = torch.zeros((T, N), dtype=torch.long, device=dev)
        self.logp = torch.zeros((T, N), device=dev)
        self.val = torch.zeros((T, N), device=dev)
        self.rew = torch.zeros((T, N), device=dev)
        self.done = torch.zeros((T, N), device=dev)


def gae(r, v, d, last_v, gamma, lam):
    T = r.shape[0]
    adv = torch.zeros_like(r)
    g = torch.zeros_like(last_v)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else v[t + 1]
        nonterm = 1.0 - d[t]
        delta = r[t] + gamma * nv * nonterm - v[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--learners", type=int, default=4)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--vlr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mb", type=int, default=512)
    p.add_argument("--ent", type=float, default=0.0005)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--beta0", type=float, default=0.1, help="anchor CE coefficient at the start")
    p.add_argument("--beta1", type=float, default=0.02, help="... at the end")
    p.add_argument("--death", type=float, default=3.0)
    p.add_argument("--leader", type=float, default=0.01)
    p.add_argument("--vpre", type=int, default=40, help="value-pretrain iterations (policy fixed)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--league", action="store_true", help="randomised league per world (LEAGUE pool) instead of the fixed endless field")
    p.add_argument("--pick", type=int, default=6, help="league entries drawn per world")
    p.add_argument("--giant-p", type=float, default=0.0, help="probability that a world has a giant opponent")
    a = p.parse_args()

    os.chdir(ROOT)
    pin_cores()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    out = os.path.join("runs", "vision", a.name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join("logs", "vision", a.name + ".jsonl"), "a")

    model, ck = load_model(a.init)
    cfg = ck["cfg"]
    model.train()
    for q in model.parameters():
        q.requires_grad_(True)
    head_params = list(model.mlp.parameters()) + list(model.head.parameters())
    hp_ids = {id(q) for q in head_params}
    rest = [q for q in model.parameters() if id(q) not in hp_ids]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=head_params, lr=a.vlr)], weight_decay=1e-4,
                            betas=(0.9, 0.99))

    if a.league:
        env = ogar_py.VecEnv(a.worlds, a.learners, opponents=[], pool=LEAGUE, pool_pick=a.pick, giant_p=a.giant_p,
                             teacher=TEACHER, px=256, threads=a.threads, seed=a.seed, warmup=0)
    else:
        env = ogar_py.VecEnv(a.worlds, a.learners, opponents=FIELD, teacher=TEACHER, px=256, threads=a.threads,
                             seed=a.seed, warmup=0, giant_p=a.giant_p)
    n = env.n
    env.set_teacher(False)
    obs = env.reset()
    info = env.info()
    ro = Roll(a.T, n)
    mass_prev = info["mass"].astype(np.float64).copy()
    act_np = np.zeros((n, 4), np.float32)
    t0 = time.time()
    budget = a.hours * 3600
    it = 0
    stats = dict(lead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0)
    last_save = 0

    def save(tag):
        torch.save(dict(cfg=cfg, model=model.state_dict(), args=vars(a), iters=it), os.path.join(out, tag + ".pt"))

    def rollout():
        nonlocal obs, info, mass_prev
        model.eval()
        for t in range(a.T):
            view = torch.from_numpy(obs["view"]).cuda()
            wide = torch.from_numpy(obs["wide"]).cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                heat, _, _, v = model(view, wide)
            logits = heat.float().flatten(1) / a.tau
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            ro.view[t], ro.wide[t] = view, wide
            ro.teacher[t] = torch.from_numpy(info["teacher"]).cuda()
            ro.act[t], ro.logp[t], ro.val[t] = idx, dist.log_prob(idx), v.float()
            act_np[:, :2] = cell_to_uv(idx).cpu().numpy()
            obs, _, done, info = env.step(act_np)
            mass = info["mass"].astype(np.float64)
            r = np.clip(np.log(np.maximum(mass, 1.0)) - np.log(np.maximum(mass_prev, 1.0)), -1.0, 1.0)
            r = np.where(done > 0.5, -a.death, r + a.leader * info["leader"])
            ro.rew[t] = torch.from_numpy(r.astype(np.float32)).cuda()
            ro.done[t] = torch.from_numpy(done.astype(np.float32)).cuda()
            mass_prev = mass
            stats["lead"] += float(info["leader"].mean())
            stats["rew"] += float(r.mean())
            stats["deaths"] += int(done.sum())
            stats["mass"] += float(mass.mean())
            stats["steps"] += 1
        view = torch.from_numpy(obs["view"]).cuda()
        wide = torch.from_numpy(obs["wide"]).cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, _, _, lv = model(view, wide)
        adv, ret = gae(ro.rew, ro.val, ro.done, lv.float(), a.gamma, a.lam)
        return adv, ret

    B = a.T * n
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        pre = it < a.vpre
        adv, ret = rollout()
        model.train()
        fv = ro.view.flatten(0, 1)
        fw = ro.wide.flatten(0, 1)
        ft = ro.teacher.flatten(0, 1)
        fa, flp = ro.act.flatten(), ro.logp.flatten()
        fadv, fret = adv.flatten(), ret.flatten()
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-6)
        beta = a.beta0 + (a.beta1 - a.beta0) * frac
        lr_scale = 1.0 if it >= a.vpre else 1.0
        for g, base in zip(opt.param_groups, (a.lr, a.vlr)):
            g["lr"] = (0.0 if pre else base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))) if g is opt.param_groups[0] \
                else base * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * frac)))
        m = dict(pl=[], vl=[], kl=[], ent=[], ce=[], clipf=[], agree=[])
        stop = False
        for ep in range(a.epochs):
            perm = torch.randperm(B, device="cuda")
            for i in range(0, B - a.mb + 1, a.mb):
                j = perm[i:i + a.mb]
                with torch.autocast("cuda", torch.bfloat16):
                    heat, _, _, v = model(fv[j], fw[j])
                heat = heat.float()
                logits = heat.flatten(1) / a.tau
                lp_all = F.log_softmax(logits, 1)
                lp = lp_all.gather(1, fa[j][:, None])[:, 0]
                ratio = torch.exp(lp - flp[j])
                if pre:
                    pl = torch.zeros((), device="cuda")
                    ent = torch.zeros((), device="cuda")
                    ce = torch.zeros((), device="cuda")
                else:
                    s1 = ratio * fadv[j]
                    s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[j]
                    pl = -torch.min(s1, s2).mean()
                    ent = -(lp_all.exp() * lp_all).sum(1).mean()
                    soft = heat_target(ft[j, 0], ft[j, 1]).flatten(1)
                    ce = -(soft * F.log_softmax(heat.flatten(1), 1)).sum(1).mean()
                vl = F.smooth_l1_loss(v.float(), fret[j])
                loss = pl - a.ent * ent + beta * ce + a.vcoef * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if pre:   # trunk frozen: only the MLP / value head learn
                    for q in rest:
                        q.grad = None
                opt.step()
                with torch.no_grad():
                    kl = (flp[j] - lp).mean()
                    m["pl"].append(pl.item()), m["vl"].append(vl.item()), m["kl"].append(kl.item())
                    m["ent"].append(ent.item()), m["ce"].append(ce.item())
                    m["clipf"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    if not pre and len(m["agree"]) < 3:
                        ang, _, _ = target_metrics(decode_target(heat), ft[j, :2])
                        m["agree"].append((ang < 15).float().mean().item())
                if not pre and kl.item() > 0.03:
                    stop = True
                    break
            if stop:
                break
        it += 1
        if it % 5 == 0:
            st = stats["steps"]
            hrs = st * DT / 3600
            rec = dict(it=it, t=round(time.time() - t0), pre=pre, lead=stats["lead"] / st, rew=stats["rew"] / st,
                       deaths_h=stats["deaths"] / n / hrs, mass=stats["mass"] / st, beta=beta,
                       **{k: float(np.mean(v)) for k, v in m.items() if v}, rate=st * n / (time.time() - t0))
            stats.update(lead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if it - last_save >= a.save_every:
            save(f"ck_{it:05d}")
            save("last")
            last_save = it
    save("final")
    save("last")
    print("done", it)


if __name__ == "__main__":
    main()
