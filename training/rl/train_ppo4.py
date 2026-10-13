"""M5: train_ppo3.py + MEMORY: the policy is recurrent (VisionNet(gru=256)); rollouts store the hidden state of every step,
the update runs BPTT over chunks of --chunk steps (the chunk's start state is the stored one, hidden states reset after a
death), and frozen snapshots of the policy carry their own hidden state per slot. Start from a fresh-GRU checkpoint made
with training/rl/mkgru.py (it plays exactly like its source net until trained).

(train_ppo3.py's docstring follows.)

M5: league PPO (train_ppo2.py) with the SPLIT action trained in.

    PYTHONPATH=logs/pymod python training/rl/train_ppo3.py --name s1 --init runs/vision/p6/final.pt --ref runs/vision/a2/final.pt --hours 2

What is new against train_ppo2.py:
  * the policy is the heat-map categorical x a Bernoulli split from the net's split logit; the executed split is the sample;
    joint log-prob in the PPO ratio. Frozen opponents still never split;
  * the env reports per learner `info["events"]` = (player cells eaten, their mass, virus pops caused on others, flag):
    flag = 1 when the rule bots' split-kill check passes on this observation (Reflex::split_kill, the reflex layer's
    teacher label). Shaped reward: + kill_bonus * min(kills, 3) + pop_bonus * pops (on top of d log mass, leader, death);
  * teacher term on the split head: pos-weighted BCE(split_logit, flag) * --split-bc (the reflex layer as the teacher);
  * for the first --split-warm iterations the executed split IS the flag (the reflex acts, the heat policy learns around it)
    and the split logit gets no PPO gradient yet.

Coordinator's diagnosis changes (M5):
  * real self-play: --frozen of the 5 learner slots per world play FROZEN COPIES of the current policy (a new snapshot
    every --snap-every iterations, --snap-keep kept), chosen with weights that favour the snapshots that are strongest
    (EMA of their slots' leader share: "the ones we lose to"); bc3 / the anchor stay as permanent members; the env's league pool
    fills the remaining opponent slots (--pick of them per world);
  * reward: d log mass + --leader (default 0.05) while #1 + --bigkill * (mass eaten from players at least as big as us / our mass)
    - --lostpen * (our cell mass eaten by other players / our mass) - death penalty;
  * the KL anchor decays to ~0 (--beta1 0.003) and lr is a bit higher (--lr 6e-5);
  * memory (stacked frames / GRU) was NOT done.

Everything below is train_ppo2.py's text. Differences to train_ppo.py:
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

import oxidish_gym

DT = 0.08

# League pool (disjoint from the held-out field plan_c34, plan2_c34, vision_bc3, hunter_virus).
LEAGUE = FIELD + [
    ("c3_4", "bots/specs/c3_4.json", 1),
    ("shoot", "bots/specs/ab/c34_shootthr.json", 1),
    ("hhunt", "bots/specs/sparring/hunter_hunt.json", 1),
    ("hreck", "bots/specs/sparring/hunter_reckless.json", 1),
    ("cm500", "bots/specs/ab/c34_cm500.json", 1),
    ("esc", "bots/specs/ab/c34_esc.json", 1),
    ("ev1", "bots/specs/ab/c34_ev1.json", 1),
    ("fs", "bots/specs/ab/c34_fs.json", 1),
    ("pd", "bots/specs/ab/c3_4_pd.json", 1),
    ("wd400", "bots/specs/ab/r300_wd400_ww800.json", 1),
    ("ww400pd", "bots/specs/ab/ww400_pd.json", 1),
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
    p.add_argument("--frozen", type=int, default=3, help="learner slots per world driven by frozen nets")
    p.add_argument("--snap-every", type=int, default=12)
    p.add_argument("--snap-keep", type=int, default=8)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--vlr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mb", type=int, default=512)
    p.add_argument("--ent", type=float, default=0.0005)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--beta0", type=float, default=0.1, help="KL anchor coefficient at the start")
    p.add_argument("--beta1", type=float, default=0.003, help="... at the end")
    p.add_argument("--death", type=float, default=3.0)
    p.add_argument("--leader", type=float, default=0.05)
    p.add_argument("--vpre", type=int, default=0, help="value-pretrain iterations (policy fixed)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--pick", type=int, default=3, help="league entries drawn per world")
    p.add_argument("--giant-p", type=float, default=0.25)
    p.add_argument("--kill-bonus", type=float, default=0.02, help="reward per player cell eaten (max 3 per decision)")
    p.add_argument("--pop-bonus", type=float, default=0.05, help="reward per virus pop caused on another player")
    p.add_argument("--split-bc", type=float, default=0.3, help="coefficient of the BCE to the reflex split-kill flag")
    p.add_argument("--split-pos-w", type=float, default=8.0, help="positive weight of that BCE (flag is rare)")
    p.add_argument("--split-warm", type=int, default=30, help="iterations in which the executed split is the flag")
    p.add_argument("--split-ent", type=float, default=0.002)
    p.add_argument("--bigkill", type=float, default=0.5, help="reward x (mass eaten from players >= us / our mass)")
    p.add_argument("--lostpen", type=float, default=1.0, help="penalty x (our mass eaten by other players / our mass)")
    p.add_argument("--chunk", type=int, default=16, help="BPTT chunk length (must divide --T)")
    p.add_argument("--split-detach", type=int, default=1, help="1: the split logit sees the CNN trunk features detached (only the head and the GRU learn from the split losses)")
    p.add_argument("--no-split", action="store_true", help="control: never split (plain PPO continuation of the init net)")
    p.add_argument("--extra-frozen", nargs="*", default=["runs/vision/bc3/final.pt"],
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
    GRU = model.gru
    if GRU:
        pass   # GRU and its projections stay in the low-lr group: at the head lr (3e-4) g_dec moved the heat map so fast that the KL early stop fired after 1-3 minibatches (g1c, g2c)
    assert a.T % a.chunk == 0
    hp_ids = {id(q) for q in head_params}
    rest = [q for q in model.parameters() if id(q) not in hp_ids]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=head_params, lr=a.vlr)], weight_decay=1e-4,
                            betas=(0.9, 0.99))

    perm_frozen = [ref] + [load_model(c)[0] for c in a.extra_frozen]
    snaps = []   # recent snapshots of the policy
    for m in perm_frozen:
        for q in m.parameters():
            q.requires_grad_(False)

    env = oxidish_gym.VecEnv(a.worlds, a.learners, opponents=[], pool=LEAGUE, pool_pick=a.pick, giant_p=a.giant_p,
                         teacher="bots/specs/plan_c34.json", px=256, threads=a.threads, seed=a.seed, warmup=0, label=False)
    n = env.n
    L = a.learners
    env.set_teacher(False)
    obs = env.reset()
    info = env.info()
    slot = np.arange(n) % L
    frozen_idx = np.nonzero(slot >= L - a.frozen)[0]
    train_idx = np.nonzero(slot < L - a.frozen)[0]
    nt = len(train_idx)
    tr = torch.from_numpy(train_idx).cuda()
    Tn = a.T
    rv = torch.zeros((Tn, nt, 2, 256, 256), dtype=torch.uint8, device="cuda")
    rw = torch.zeros((Tn, nt, 2, 128, 128), dtype=torch.uint8, device="cuda")
    ract = torch.zeros((Tn, nt), dtype=torch.long, device="cuda")
    rspl = torch.zeros((Tn, nt), device="cuda")      # executed split (0/1)
    rflag = torch.zeros((Tn, nt), device="cuda")     # reflex split-kill flag on that observation
    rsm = torch.zeros((Tn, nt), device="cuda")       # 1 where the split was sampled by the policy (PPO term applies)
    rlogps = torch.zeros((Tn, nt), device="cuda")
    rh = torch.zeros((Tn, nt, max(GRU, 1)), device="cuda")      # hidden state going into each step
    h_cur = torch.zeros((nt, max(GRU, 1)), device="cuda")
    h_fro = torch.zeros((len(frozen_idx), max(GRU, 1)), device="cuda")
    rlogp = torch.zeros((Tn, nt), device="cuda")
    rval = torch.zeros((Tn, nt), device="cuda")
    rrew = torch.zeros((Tn, nt), device="cuda")
    rdone = torch.zeros((Tn, nt), device="cuda")
    mass_prev = info["mass"].astype(np.float64).copy()
    act_np = np.zeros((n, 4), np.float32)
    rng = np.random.default_rng(a.seed)
    assign = np.zeros(len(frozen_idx), int)
    t0 = time.time()
    budget = a.hours * 3600
    it = 0
    stats = dict(lead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0, flead=0.0, kills=0.0, kmass=0.0, pops=0.0, splits=0.0, flags=0.0)
    last_save = 0
    nsamp = 0

    def pool():
        return perm_frozen + snaps

    strength = {}   # id(net) -> EMA of the leader share of the slots that played it

    def redraw():
        pl = pool()
        w = np.array([0.4 if i < len(perm_frozen) else 1.0 + 8.0 * strength.get(id(m), 0.1) for i, m in enumerate(pl)])
        assign[:] = rng.choice(len(pl), len(frozen_idx), p=w / w.sum())
        h_fro.zero_()

    redraw()

    def save(tag):
        torch.save(dict(cfg=cfg, model=model.state_dict(), args=vars(a), iters=it), os.path.join(out, tag + ".pt"))

    def frozen_actions(view, wide):
        """Greedy actions of the frozen nets for the frozen slots (grouped by pool member)."""
        pl = pool()
        for k, net in enumerate(pl):
            sel = np.nonzero(assign == k)[0]
            if len(sel) == 0:
                continue
            ids = frozen_idx[sel]
            gi = torch.from_numpy(ids).cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                if getattr(net, "gru", 0):
                    si = torch.from_numpy(sel).cuda()
                    heat, _, _, _, hn = net.forward_rec(view[gi], wide[gi], h_fro[si])
                    h_fro[si] = hn.float()
                else:
                    heat, _, _, _ = net(view[gi], wide[gi])
            act_np[ids, :2] = decode_target(heat.float()).cpu().numpy()

    def rollout():
        nonlocal obs, info, mass_prev, h_cur
        model.eval()
        warm = it < a.split_warm
        for t in range(Tn):
            view = torch.from_numpy(obs["view"]).cuda()
            wide = torch.from_numpy(obs["wide"]).cuda()
            vt, wt = view[tr], wide[tr]
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                if GRU:
                    heat, sl, _, v, hn = model.forward_rec(vt, wt, h_cur)
                else:
                    heat, sl, _, v = model(vt, wt)
                    hn = h_cur
            rh[t] = h_cur
            sdist = torch.distributions.Bernoulli(logits=sl.float())
            flag_np = info["events"][train_idx, 3]
            if not torch.isfinite(sl).all():
                raise SystemExit("non-finite split logits: model diverged")
            flag = torch.from_numpy(flag_np.astype(np.float32)).cuda()
            spl = torch.zeros_like(flag) if a.no_split else (flag.clone() if warm else sdist.sample())
            logits = heat.float().flatten(1) / a.tau
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            rv[t], rw[t] = vt, wt
            rspl[t], rflag[t], rsm[t], rlogps[t] = spl, flag, 0.0 if warm else 1.0, sdist.log_prob(spl)
            ract[t], rlogp[t], rval[t] = idx, dist.log_prob(idx), v.float()
            act_np[train_idx, :2] = cell_to_uv(idx).cpu().numpy()
            act_np[train_idx, 2] = spl.cpu().numpy()
            frozen_actions(view, wide)
            obs, _, done, info = env.step(act_np)
            mass = info["mass"].astype(np.float64)
            r = np.clip(np.log(np.maximum(mass, 1.0)) - np.log(np.maximum(mass_prev, 1.0)), -1.0, 1.0)
            ev = info["events"]
            mp = np.maximum(mass_prev, 1.0)
            r = r + a.kill_bonus * np.minimum(ev[:, 0], 3.0) + a.pop_bonus * ev[:, 2]
            r = r + a.bigkill * np.minimum(ev[:, 4] / mp, 1.0) - a.lostpen * np.minimum(ev[:, 5] / mp, 1.0)
            pl_ = pool()
            for k_, net_ in enumerate(pl_):
                sel = assign == k_
                if sel.any():
                    lead_k = float(info["leader"][frozen_idx[sel]].mean())
                    strength[id(net_)] = 0.98 * strength.get(id(net_), lead_k) + 0.02 * lead_k
            r = np.where(done > 0.5, -a.death, r + a.leader * info["leader"])
            rrew[t] = torch.from_numpy(r[train_idx].astype(np.float32)).cuda()
            rdone[t] = torch.from_numpy(done[train_idx].astype(np.float32)).cuda()
            h_cur = hn.float() * (1.0 - rdone[t])[:, None]
            h_fro.mul_(torch.from_numpy(1.0 - done[frozen_idx].astype(np.float32)).cuda()[:, None])
            mass_prev = mass
            stats["lead"] += float(info["leader"][train_idx].mean())
            stats["flead"] += float(info["leader"][frozen_idx].mean())
            stats["rew"] += float(r[train_idx].mean())
            stats["deaths"] += int(done[train_idx].sum())
            stats["mass"] += float(mass[train_idx].mean())
            stats["kills"] += float(ev[train_idx, 0].mean())
            stats["kmass"] += float(ev[train_idx, 1].mean())
            stats["pops"] += float(ev[train_idx, 2].mean())
            stats["splits"] += float(spl.mean())
            stats["flags"] += float(flag.mean())
            stats["steps"] += 1
        view = torch.from_numpy(obs["view"]).cuda()[tr]
        wide = torch.from_numpy(obs["wide"]).cuda()[tr]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            lv = (model.forward_rec(view, wide, h_cur)[3] if GRU else model(view, wide)[3])
        return gae(rrew, rval, rdone, lv.float(), a.gamma, a.lam)

    B = Tn * nt
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        pre = it < a.vpre
        tr0 = time.time()
        adv, ret = rollout()
        t_roll = time.time() - tr0
        model.train()
        fadv = adv
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-6)
        beta = a.beta0 + (a.beta1 - a.beta0) * frac
        cosf = 0.5 * (1 + math.cos(math.pi * frac))
        opt.param_groups[0]["lr"] = 0.0 if pre else a.lr * (0.1 + 0.9 * cosf)
        opt.param_groups[1]["lr"] = a.vlr * (0.3 + 0.7 * cosf)
        m = dict(pl=[], vl=[], kl=[], ent=[], akl=[], clipf=[], agree=[], sbc=[], sent=[], psplit=[])
        stop = False
        tt0 = time.time()
        Lc = a.chunk
        P = max(1, a.mb // Lc)
        n_pairs = (Tn // Lc) * nt
        ar = torch.arange(Lc, device="cuda")[None, :]
        for ep in range(a.epochs):
            perm = torch.randperm(n_pairs, device="cuda")
            for i in range(0, n_pairs - P + 1, P):
                pj = perm[i:i + P]
                ci, ni = pj // nt, pj % nt
                ti = ci[:, None] * Lc + ar                       # [P, Lc] time index
                nn_ = ni[:, None]

                def G(x):
                    return x[ti, nn_].flatten(0, 1)

                vv, ww = G(rv), G(rw)
                fa, flp, fspl, fflag, fsm, flps, fadvj, fretj = G(ract), G(rlogp), G(rspl), G(rflag), G(rsm), G(rlogps), G(fadv), G(ret)
                with torch.autocast("cuda", torch.bfloat16):
                    if GRU:
                        e = model.encode(vv, ww)
                        feat = e["feat"].float().view(P, Lc, -1)
                        h = rh[ti[:, 0], ni]
                        gs = []
                        for l_ in range(Lc):
                            if l_ > 0:
                                h = h * (1.0 - rdone[ti[:, l_ - 1], ni])[:, None]
                            with torch.autocast("cuda", enabled=False):
                                h = model.cell(feat[:, l_].float(), h.float())
                            gs.append(h)
                        g = torch.stack(gs, 1).flatten(0, 1)
                        heat = model.decode(e, g)
                        sl, _, v = model.heads(e["feat"], g)
                        if a.split_detach:
                            sl = model.heads(e["feat"].detach(), g)[0]
                    else:
                        heat, sl, _, v = model(vv, ww)
                    with torch.no_grad():
                        rheat = ref(vv, ww)[0]
                heat = heat.float()
                lp_all = F.log_softmax(heat.flatten(1) / a.tau, 1)
                lp = lp_all.gather(1, fa[:, None])[:, 0]
                if os.environ.get("PPO_DEBUG_KL") and ep == 0 and i == 0:
                    d_ = (flp - lp).detach().view(P, Lc).abs()
                    print("DEBUGKL it", it, "abs diff per chunk position", [round(x, 3) for x in d_.mean(0).tolist()], flush=True)
                sd = torch.distributions.Bernoulli(logits=sl.float())
                lps = sd.log_prob(fspl) * fsm
                lp_joint = lp + lps
                ratio = torch.exp(lp_joint - (flp + flps * fsm))
                sbc = F.binary_cross_entropy_with_logits(sl.float(), fflag, pos_weight=torch.tensor(a.split_pos_w, device="cuda"))
                sent = sd.entropy().mean()
                if pre:
                    pl = ent = akl = torch.zeros((), device="cuda")
                else:
                    s1 = ratio * fadvj
                    s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * fadvj
                    pl = -torch.min(s1, s2).mean()
                    ent = -(lp_all.exp() * lp_all).sum(1).mean()
                    lq = F.log_softmax(heat.flatten(1), 1)
                    lr_ = F.log_softmax(rheat.float().flatten(1), 1)
                    akl = (lr_.exp() * (lr_ - lq)).sum(1).mean()      # KL(ref || pi), tau = 1
                vl = F.smooth_l1_loss(v.float(), fretj)
                loss = pl - a.ent * ent - a.split_ent * sent + beta * akl + a.vcoef * vl + a.split_bc * sbc
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):
                    opt.zero_grad(set_to_none=True)
                    continue
                if pre:
                    for q in rest:
                        q.grad = None
                opt.step()
                nsamp += a.mb
                with torch.no_grad():
                    kl = ((flp + flps * fsm) - lp_joint).mean()
                    m["sbc"].append(sbc.item()), m["sent"].append(sent.item()), m["psplit"].append(torch.sigmoid(sl.float()).mean().item())
                    m["pl"].append(pl.item()), m["vl"].append(vl.item()), m["kl"].append(kl.item())
                    m["ent"].append(ent.item()), m["akl"].append(akl.item())
                    m["clipf"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    if not pre and len(m["agree"]) < 3:
                        ang = (decode_target(heat) - decode_target(rheat.float()))
                        m["agree"].append((ang.norm(dim=1) < 12).float().mean().item())
                if not pre and kl.item() > 0.03:
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
                       roll_dps=Tn * n / max(t_roll, 1e-9), npool=len(pool()),
                       **{k: float(np.mean(v)) for k, v in m.items() if v})
            rec.update(kills=stats["kills"] / st, kmass=stats["kmass"] / st, pops=stats["pops"] / st,
                       splits=stats["splits"] / st, flags=stats["flags"] / st)
            stats.update(lead=0.0, flead=0.0, rew=0.0, deaths=0, steps=0, mass=0.0, kills=0.0, kmass=0.0, pops=0.0, splits=0.0, flags=0.0)
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
