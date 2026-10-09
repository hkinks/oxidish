"""M7: league PPO with the hunter reflex in the loop (docs/vision-bot.md, M7). Based on train_ppo2.py.

    python rl/vision/train_ppo5.py --name m7a --init runs/vision/p6/final.pt --ref runs/vision/p6/final.pt --hours 2

Differences to train_ppo2.py:
  * the learners (and the frozen "hunter" copies) run the hunter reflex inside the env (`ogar_py.VecEnv(hunter=genes)`,
    `set_hunter`): while its commit phase is active the env replaces the net's aim / split / eject; `info["hunter"]` marks
    those observations and their policy-gradient and entropy terms are masked out (off-policy: the net did not act);
    the value head still trains on them;
  * stronger league: planhv (plan_hv_commit_only) and c34h (hv_commit_only) are in the pool; frozen slots are bc3, the
    anchor, p6, p6 + hunter and snapshots of the policy (+ hunter), sampled with the strong ones more often;
  * reward (M5): d log mass + --leader while #1 + --bigkill x (mass eaten from players >= us) - --lostpen x (own mass eaten)
    - death; the KL anchor to --ref decays from --beta0 to --beta1; the KL early stop needs --min-mb minibatches first;
  * no split (the hunter does the safe splits).

Differences of train_ppo2.py to train_ppo.py:
  * the env runs with label=False (no planner teacher, ~3x faster env);
  * anchor = KL(ref || pi) to a frozen BC net evaluated in the same minibatch (no teacher labels);
  * the last `--frozen` learner slots of every world are driven by FROZEN nets (bc3, the BC anchor and snapshots of the
    policy, greedy decode), computed on the GPU in the same step; they produce no training samples;
  * opponents: randomised league pool per world (`--league`) and an optional giant opponent.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import FIELD, ROOT, pin_cores
from evalval import load_model
from model import decode_target

import ogar_py

DT = 0.08

# League pool (disjoint from the held-out field plan_c34, plan2_c34, vision_bc3, apos, hunter_virus).
HUNT_GENES = "bot/drill/hv_commit_only.json"
LEAGUE = FIELD + [
    ("planhv", "bot/drill/plan_hv_commit_only.json", 1),
    ("planhv2", "bot/drill/plan_hv_commit_only.json", 1),
    ("c34h", "bot/drill/hv_commit_only.json", 1),
    ("c34h2", "bot/drill/hv_commit_only.json", 1),
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
    p.add_argument("--ref", required=True, help="frozen BC net for the KL anchor (and a permanent frozen opponent)")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--learners", type=int, default=5)
    p.add_argument("--frozen", type=int, default=2, help="learner slots per world driven by frozen nets")
    p.add_argument("--snap-every", type=int, default=60)
    p.add_argument("--snap-keep", type=int, default=3)
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
    p.add_argument("--beta0", type=float, default=0.2, help="KL anchor coefficient at the start")
    p.add_argument("--beta1", type=float, default=0.01, help="... at the end")
    p.add_argument("--klstop", type=float, default=0.05, help="KL early stop of the iteration")
    p.add_argument("--min-mb", type=int, default=4, help="minibatches before the KL early stop may fire")
    p.add_argument("--bigkill", type=float, default=0.5, help="reward x (mass eaten from players >= us / our mass)")
    p.add_argument("--lostpen", type=float, default=1.0, help="penalty x (our mass eaten by other players / our mass)")
    p.add_argument("--no-hunter", action="store_true", help="control: no hunter reflex in the loop")
    p.add_argument("--death", type=float, default=3.0)
    p.add_argument("--leader", type=float, default=0.05)
    p.add_argument("--vpre", type=int, default=0, help="value-pretrain iterations (policy fixed)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--pick", type=int, default=5, help="league entries drawn per world")
    p.add_argument("--giant-p", type=float, default=0.25)
    p.add_argument("--extra-frozen", nargs="*", default=["runs/vision/bc3/final.pt", "runs/vision/p6/final.pt"],
                   help="more permanent frozen opponents (checkpoints)")
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
    ref, _ = load_model(a.ref)
    for q in ref.parameters():
        q.requires_grad_(False)
    head_params = list(model.mlp.parameters()) + list(model.head.parameters())
    hp_ids = {id(q) for q in head_params}
    rest = [q for q in model.parameters() if id(q) not in hp_ids]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=head_params, lr=a.vlr)], weight_decay=1e-4,
                            betas=(0.9, 0.99))

    perm_frozen = [ref] + [load_model(c)[0] for c in a.extra_frozen]
    snaps = []   # recent snapshots of the policy
    for m in perm_frozen:
        for q in m.parameters():
            q.requires_grad_(False)
    # pool members as (net, runs the hunter): the permanent nets play without it, p6 also with it, snapshots with it
    # (they are the learner's own policy, which plays with the hunter)
    hunt = not a.no_hunter
    p6net = perm_frozen[-1]

    env = ogar_py.VecEnv(a.worlds, a.learners, opponents=[], pool=LEAGUE, pool_pick=a.pick, giant_p=a.giant_p,
                         teacher="bot/plan_c34.json", px=256, threads=a.threads, seed=a.seed, warmup=0, label=False,
                         hunter=HUNT_GENES if hunt else None)
    n = env.n
    L = a.learners
    env.set_teacher(False)
    slot = np.arange(n) % L
    frozen_idx = np.nonzero(slot >= L - a.frozen)[0]
    train_idx = np.nonzero(slot < L - a.frozen)[0]
    nt = len(train_idx)
    hmask = np.zeros(n, bool)
    if hunt:
        hmask[train_idx] = True
        env.set_hunter(hmask.tolist())
    obs = env.reset()
    info = env.info()
    tr = torch.from_numpy(train_idx).cuda()
    Tn = a.T
    rv = torch.zeros((Tn, nt, 2, 256, 256), dtype=torch.uint8, device="cuda")
    rw = torch.zeros((Tn, nt, 2, 128, 128), dtype=torch.uint8, device="cuda")
    ract = torch.zeros((Tn, nt), dtype=torch.long, device="cuda")
    rlogp = torch.zeros((Tn, nt), device="cuda")
    rval = torch.zeros((Tn, nt), device="cuda")
    rrew = torch.zeros((Tn, nt), device="cuda")
    rdone = torch.zeros((Tn, nt), device="cuda")
    rmask = torch.ones((Tn, nt), device="cuda")      # 1 = the net acted, 0 = the hunter overrode this decision
    mass_prev = info["mass"].astype(np.float64).copy()
    act_np = np.zeros((n, 4), np.float32)
    rng = np.random.default_rng(a.seed)
    assign = np.zeros(len(frozen_idx), int)
    t0 = time.time()
    budget = a.hours * 3600
    it = 0
    stats = dict(lead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0, flead=0.0, hov=0.0, hfroz=0.0)
    last_save = 0
    nsamp = 0

    def pool():
        return perm_frozen + snaps

    def members():
        """(net, hunter) frozen-slot candidates; p6 + hunter and the snapshots (+ hunter) are listed twice (drawn more often)."""
        m = [(net, False) for net in perm_frozen]
        if hunt:
            m += [(p6net, True), (p6net, True)]
            m += [(s_, True) for s_ in snaps for _ in range(2)]
        else:
            m += [(s_, False) for s_ in snaps]
        return m

    def redraw():
        mem = members()
        assign[:] = rng.integers(0, len(mem), len(frozen_idx))
        if hunt:
            hm = hmask.copy()
            hm[frozen_idx] = [mem[k][1] for k in assign]
            env.set_hunter(hm.tolist())

    redraw()

    def save(tag):
        torch.save(dict(cfg=cfg, model=model.state_dict(), args=vars(a), iters=it), os.path.join(out, tag + ".pt"))

    def frozen_actions(view, wide):
        """Greedy actions of the frozen nets for the frozen slots (grouped by pool member)."""
        mem = members()
        for k, (net, _) in enumerate(mem):
            sel = np.nonzero(assign == k)[0]
            if len(sel) == 0:
                continue
            ids = frozen_idx[sel]
            gi = torch.from_numpy(ids).cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                heat, _, _, _ = net(view[gi], wide[gi])
            act_np[ids, :2] = decode_target(heat.float()).cpu().numpy()

    def rollout():
        nonlocal obs, info, mass_prev
        model.eval()
        for t in range(Tn):
            view = torch.from_numpy(obs["view"]).cuda()
            wide = torch.from_numpy(obs["wide"]).cuda()
            vt, wt = view[tr], wide[tr]
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                heat, _, _, v = model(vt, wt)
            logits = heat.float().flatten(1) / a.tau
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            rv[t], rw[t] = vt, wt
            rmask[t] = 1.0 - torch.from_numpy(info["hunter"][train_idx].astype(np.float32)).cuda()
            stats["hov"] += float(info["hunter"][train_idx].mean())
            stats["hfroz"] += float(info["hunter"][frozen_idx].mean())
            ract[t], rlogp[t], rval[t] = idx, dist.log_prob(idx), v.float()
            act_np[train_idx, :2] = cell_to_uv(idx).cpu().numpy()
            frozen_actions(view, wide)
            obs, _, done, info = env.step(act_np)
            mass = info["mass"].astype(np.float64)
            r = np.clip(np.log(np.maximum(mass, 1.0)) - np.log(np.maximum(mass_prev, 1.0)), -1.0, 1.0)
            ev = info["events"]
            mp = np.maximum(mass_prev, 1.0)
            r = r + a.bigkill * np.minimum(ev[:, 4] / mp, 1.0) - a.lostpen * np.minimum(ev[:, 5] / mp, 1.0)
            r = np.where(done > 0.5, -a.death, r + a.leader * info["leader"])
            rrew[t] = torch.from_numpy(r[train_idx].astype(np.float32)).cuda()
            rdone[t] = torch.from_numpy(done[train_idx].astype(np.float32)).cuda()
            mass_prev = mass
            stats["lead"] += float(info["leader"][train_idx].mean())
            stats["flead"] += float(info["leader"][frozen_idx].mean())
            stats["rew"] += float(r[train_idx].mean())
            stats["deaths"] += int(done[train_idx].sum())
            stats["mass"] += float(mass[train_idx].mean())
            stats["steps"] += 1
        view = torch.from_numpy(obs["view"]).cuda()[tr]
        wide = torch.from_numpy(obs["wide"]).cuda()[tr]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, _, _, lv = model(view, wide)
        return gae(rrew, rval, rdone, lv.float(), a.gamma, a.lam)

    B = Tn * nt
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        pre = it < a.vpre
        tr0 = time.time()
        adv, ret = rollout()
        t_roll = time.time() - tr0
        model.train()
        fv, fw = rv.flatten(0, 1), rw.flatten(0, 1)
        fa, flp = ract.flatten(), rlogp.flatten()
        fadv, fret = adv.flatten(), ret.flatten()
        fmask = rmask.flatten()
        sel_ = fmask > 0.5
        fadv = (fadv - fadv[sel_].mean()) / (fadv[sel_].std() + 1e-6)
        beta = a.beta0 + (a.beta1 - a.beta0) * frac
        cosf = 0.5 * (1 + math.cos(math.pi * frac))
        opt.param_groups[0]["lr"] = 0.0 if pre else a.lr * (0.1 + 0.9 * cosf)
        opt.param_groups[1]["lr"] = a.vlr * (0.3 + 0.7 * cosf)
        m = dict(pl=[], vl=[], kl=[], ent=[], akl=[], clipf=[], agree=[])
        stop = False
        tt0 = time.time()
        for ep in range(a.epochs):
            perm = torch.randperm(B, device="cuda")
            for i in range(0, B - a.mb + 1, a.mb):
                j = perm[i:i + a.mb]
                with torch.autocast("cuda", torch.bfloat16):
                    heat, _, _, v = model(fv[j], fw[j])
                    with torch.no_grad():
                        rheat = ref(fv[j], fw[j])[0]
                heat = heat.float()
                lp_all = F.log_softmax(heat.flatten(1) / a.tau, 1)
                lp = lp_all.gather(1, fa[j][:, None])[:, 0]
                ratio = torch.exp(lp - flp[j])
                if pre:
                    pl = ent = akl = torch.zeros((), device="cuda")
                else:
                    wm = fmask[j]
                    wsum = wm.sum().clamp(min=1.0)
                    s1 = ratio * fadv[j]
                    s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[j]
                    pl = -(torch.min(s1, s2) * wm).sum() / wsum
                    ent = -((lp_all.exp() * lp_all).sum(1) * wm).sum() / wsum
                    lq = F.log_softmax(heat.flatten(1), 1)
                    lr_ = F.log_softmax(rheat.float().flatten(1), 1)
                    akl = (lr_.exp() * (lr_ - lq)).sum(1).mean()      # KL(ref || pi), tau = 1
                vl = F.smooth_l1_loss(v.float(), fret[j])
                loss = pl - a.ent * ent + beta * akl + a.vcoef * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if pre:
                    for q in rest:
                        q.grad = None
                opt.step()
                nsamp += a.mb
                with torch.no_grad():
                    kl = ((flp[j] - lp) * fmask[j]).sum() / fmask[j].sum().clamp(min=1.0)
                    m["pl"].append(pl.item()), m["vl"].append(vl.item()), m["kl"].append(kl.item())
                    m["ent"].append(ent.item()), m["akl"].append(akl.item())
                    m["clipf"].append((((ratio - 1).abs() > a.clip).float() * fmask[j]).sum().item() / max(fmask[j].sum().item(), 1.0))
                    if not pre and len(m["agree"]) < 3:
                        ang = (decode_target(heat) - decode_target(rheat.float()))
                        m["agree"].append((ang.norm(dim=1) < 12).float().mean().item())
                if not pre and kl.item() > a.klstop and len(m["kl"]) >= a.min_mb:
                    stop = True
                    break
            if stop:
                break
        t_train = time.time() - tt0
        it += 1
        if it % a.snap_every == 0:
            s = copy.deepcopy(model).eval()
            for q in s.parameters():
                q.requires_grad_(False)
            snaps.append(s)
            del snaps[:-a.snap_keep]
            redraw()
        if it % 5 == 0:
            st = stats["steps"]
            hrs = st * DT / 3600
            rec = dict(it=it, t=round(time.time() - t0), pre=pre, lead=stats["lead"] / st, flead=stats["flead"] / st,
                       rew=stats["rew"] / st, deaths_h=stats["deaths"] / nt / hrs, mass=stats["mass"] / st, beta=beta,
                       t_roll=t_roll, t_train=t_train, train_sps=B * a.epochs / max(t_train, 1e-9),
                       roll_dps=Tn * n / max(t_roll, 1e-9), npool=len(pool()), nmb=len(m["kl"]),
                       hov=stats["hov"] / st, hfroz=stats["hfroz"] / st,
                       **{k: float(np.mean(v)) for k, v in m.items() if v})
            stats.update(lead=0.0, flead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0, hov=0.0, hfroz=0.0)
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
