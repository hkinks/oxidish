"""Behaviour cloning + DAgger on the oxidish_gym VecEnv, streaming (no shards). Run from the repo root:

    python training/rl/train_bc.py --name bc1 --hours 1.5

Collector thread: steps the env (teacher labels come with every observation), runs the student on the learners it
drives (DAgger: per-learner `set_teacher`, the fraction of student-driven learners ramps up), and pushes the
current frames + teacher labels into a GPU ring. Trainer (main thread): samples the ring uniformly, cross-entropy
on a soft Gaussian heat-map target + weighted BCE for split / eject. Evaluates an EMA copy on the fixed held-out set.
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import FIELD, ROOT, TEACHER, VAL_DIR, pin_cores, target_metrics
from evalval import evaluate, load_windows
from model import VisionNet, decode_target, heat_target
from store import FrameStore, PosStore

import oxidish_gym


def augment(view, wide, tgt, px=256):
    """Dihedral symmetry (8 transforms, one per chunk of the batch): the game is rotation/flip invariant."""
    B = view.shape[0]
    outs_v, outs_w, outs_t = [], [], []
    for k, idx in enumerate(torch.arange(B, device=view.device).chunk(8)):
        v, w, t = view[idx], wide[idx], tgt[idx].clone()
        if k & 1:   # flip u
            v, w = v.flip(3), w.flip(3)
            t[:, 0] = px - t[:, 0]
        if k & 2:   # flip v
            v, w = v.flip(2), w.flip(2)
            t[:, 1] = px - t[:, 1]
        if k & 4:   # transpose
            v, w = v.transpose(2, 3), w.transpose(2, 3)
            t = t[:, [1, 0]]
        outs_v.append(v), outs_w.append(w), outs_t.append(t)
    return torch.cat(outs_v), torch.cat(outs_w), torch.cat(outs_t)


class Shared:
    def __init__(self):
        self.stop = False
        self.trained = 0          # samples consumed by the trainer
        self.collected = 0        # samples pushed to the ring
        self.p_student = 0.0
        self.split_thr = 0.5
        self.lock = threading.Lock()
        self.infer_sd = None      # latest EMA weights (state dict on GPU)
        self.infer_ver = 0
        self.env_rate = 0.0
        self.deaths = 0
        self.danger_flagged = 0
        self.error = None


def collector(a, sh, store, pos, infer, danger=None):
    try:
        env = oxidish_gym.VecEnv(a.worlds, a.learners, opponents=FIELD, teacher=TEACHER, px=256, threads=a.threads,
                             seed=a.seed, warmup=0)
        n = env.n
        assert n == store.N
        rng = np.random.default_rng(a.seed)
        obs = env.reset()
        info = env.info()
        first = torch.ones(n, dtype=torch.bool, device="cuda")
        teacher_mask = np.ones(n, bool)
        prev_student = torch.zeros(n, dtype=torch.bool, device="cuda")
        env.set_teacher(True)
        ver = -1
        t0, steps = time.time(), 0
        ring_samples = store.T * store.N
        H = a.danger_h
        mass_hist = np.zeros((H + 1, n), np.float32)
        t_push = 0
        while not sh.stop:
            while not sh.stop and sh.collected > sh.trained * a.ratio + 0.3 * ring_samples:
                time.sleep(0.002)
            if ver != sh.infer_ver and sh.infer_sd is not None:
                infer.load_state_dict(sh.infer_sd)
                ver = sh.infer_ver
            view = torch.from_numpy(obs["view"]).cuda()
            wide = torch.from_numpy(obs["wide"]).cuda()
            teacher = torch.from_numpy(info["teacher"]).cuda()
            store.push(view[:, 0], wide[:, 0], teacher, first, prev_student)
            hit = ((teacher[:, 2] > 0.5) & ~first).nonzero().flatten()
            if len(hit):
                pos.add(view[hit], wide[hit], teacher[hit])
            sh.collected += n
            if danger is not None:
                mass_hist[t_push % (H + 1)] = info["mass"]
                t_push += 1
                if t_push > H:
                    s0 = mass_hist[(t_push - 1 - H) % (H + 1)]
                    later = np.stack([mass_hist[(t_push - 1 - H + k) % (H + 1)] for k in range(1, H + 1)])
                    flag = (later.min(0) < a.danger_loss * s0) & (s0 >= a.danger_min_mass)
                    idx = torch.from_numpy(np.nonzero(flag)[0]).cuda()
                    if len(idx):
                        row = (store.t - 1 - H) % store.T
                        rows = torch.full_like(idx, row)
                        v, w, tch, _ = store.gather(rows, idx)
                        danger.add(v, w, tch)
                        sh.danger_flagged += len(idx)
            # per-learner driver switching (segments of ~150 decisions)
            flip = rng.random(n) < 1 / 150
            want_student = rng.random(n) < sh.p_student
            teacher_mask = np.where(flip, ~want_student, teacher_mask)
            if sh.p_student == 0:
                teacher_mask[:] = True
            act = info["teacher"].copy()
            if sh.p_student > 0 and ver >= 0:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    heat, s, e, _ = infer(view, wide)
                uv = decode_target(heat.float())
                sa = torch.zeros((n, 4), device="cuda")
                sa[:, :2] = uv
                sa[:, 2] = (torch.sigmoid(s.float()) > sh.split_thr).float()
                sa[:, 3] = (torch.sigmoid(e.float()) > 0.5).float()
                sa = sa.cpu().numpy()
                act = np.where(teacher_mask[:, None], act, sa)
            else:
                teacher_mask[:] = True
            env.set_teacher(teacher_mask.tolist())
            obs, _, done, info = env.step(np.ascontiguousarray(act, dtype=np.float32))
            d = torch.from_numpy(done).cuda().bool()
            first = d
            prev_student = torch.from_numpy(~teacher_mask).cuda()
            sh.deaths += int(done.sum())
            steps += 1
            if steps % 50 == 0:
                sh.env_rate = steps * n / (time.time() - t0)
    except Exception as ex:  # surface to the main thread
        import traceback
        traceback.print_exc()
        sh.error = ex
        sh.stop = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--learners", type=int, default=4)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--ring", type=int, default=384, help="ring length in decisions per learner")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sigma", type=float, default=1.0, help="heat target std in cells")
    p.add_argument("--pos-weight", type=float, default=8.0, help="split BCE positive weight")
    p.add_argument("--ratio", type=float, default=1.0, help="new samples collected per sample trained")
    p.add_argument("--pmax", type=float, default=0.7, help="DAgger: final fraction of student-driven learners")
    p.add_argument("--d0", type=float, default=0.05, help="DAgger ramp start, fraction of the time budget")
    p.add_argument("--d1", type=float, default=0.5, help="DAgger ramp end")
    p.add_argument("--pos-frac", type=float, default=0.03, help="fraction of each batch drawn from the split-event store")
    p.add_argument("--pos-cap", type=int, default=16000)
    p.add_argument("--danger-cap", type=int, default=0, help="size of the danger / death-precursor store (0 = off)")
    p.add_argument("--danger-frac", type=float, default=0.15, help="fraction of each batch drawn from it")
    p.add_argument("--danger-h", type=int, default=30, help="decisions before a loss that count as precursors")
    p.add_argument("--danger-loss", type=float, default=0.7, help="mass falls below this fraction within danger-h")
    p.add_argument("--danger-min-mass", type=float, default=40.0)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--eval-min", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--init", default=None, help="checkpoint to start from")
    a = p.parse_args()

    os.chdir(ROOT)
    pin_cores()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    out = os.path.join("runs", "vision", a.name)
    os.makedirs(out, exist_ok=True)
    os.makedirs("logs/vision", exist_ok=True)
    logf = open(os.path.join("logs", "vision", a.name + ".jsonl"), "a")

    cfg = dict(width=a.width)
    model = VisionNet(**cfg).cuda().to(memory_format=torch.channels_last)
    ema = VisionNet(**cfg).cuda().to(memory_format=torch.channels_last)
    infer = VisionNet(**cfg).cuda().to(memory_format=torch.channels_last).eval()
    if a.init:
        sd = torch.load(a.init, map_location="cuda", weights_only=False)["model"]
        model.load_state_dict(sd)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    decay, no_decay = [], []
    for n_, q in model.named_parameters():
        (no_decay if q.ndim < 2 else decay).append(q)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=no_decay, weight_decay=0)], lr=a.lr,
                            betas=(0.9, 0.99))

    print("loading validation set", flush=True)
    val_main = load_windows(os.path.join(VAL_DIR, "win_*.npz"))
    ev_path = "runs/vision/val_events/events.npz"
    val_ev = load_windows(ev_path) if os.path.exists(ev_path) else None

    N = a.worlds * a.learners
    store = FrameStore(a.ring, N)
    pos = PosStore(a.pos_cap)
    danger = PosStore(a.danger_cap) if a.danger_cap else None
    sh = Shared()
    th = threading.Thread(target=collector, args=(a, sh, store, pos, infer, danger), daemon=True)
    th.start()
    budget = a.hours * 3600
    t_start = time.time()
    next_eval = t_start + a.eval_min * 60
    best = -1.0
    step = 0
    acc = {}
    t_log = time.time()
    warm_rows = min(a.ring, 64)
    while store.filled < warm_rows and not sh.stop:
        time.sleep(0.2)
    print(f"collector warm ({store.filled} rows); training", flush=True)
    pos_w = torch.tensor(a.pos_weight, device="cuda")

    def save(tag, metrics=None):
        torch.save(dict(cfg=cfg, model=ema.state_dict(), args=vars(a), samples=sh.trained, metrics=metrics,
                        split_thr=sh.split_thr), os.path.join(out, tag + ".pt"))

    while not sh.stop:
        el = time.time() - t_start
        frac = el / budget
        if frac >= 1.0:
            break
        warm = min(1.0, (step + 1) / 300)
        lr = a.lr * warm * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        sh.p_student = a.pmax * min(1.0, max(0.0, (frac - a.d0) / max(a.d1 - a.d0, 1e-6)))

        k_pos = int(round(a.pos_frac * a.batch)) if pos.filled >= 64 else 0
        k_dan = int(round(a.danger_frac * a.batch)) if danger is not None and danger.filled >= 1000 else 0
        rows, cols = store.sample_index(a.batch - k_pos - k_dan)
        view, wide, teacher, stud = store.gather(rows, cols)
        if k_pos:
            pv, pw, pt = pos.sample(k_pos)
            view, wide, teacher = torch.cat([view, pv]), torch.cat([wide, pw]), torch.cat([teacher, pt])
        if k_dan:
            pv, pw, pt = danger.sample(k_dan)
            view, wide, teacher = torch.cat([view, pv]), torch.cat([wide, pw]), torch.cat([teacher, pt])
        if not a.no_aug:   # chunks keep the sample order, so teacher[:, 2:] still lines up
            view, wide, tgt = augment(view, wide, teacher[:, :2])
        else:
            tgt = teacher[:, :2]
        with torch.autocast("cuda", torch.bfloat16):
            heat, s, e, _ = model(view, wide)
        heat = heat.float()
        logp = F.log_softmax(heat.flatten(1), 1)
        soft = heat_target(tgt[:, 0], tgt[:, 1], sigma=a.sigma).flatten(1)
        l_heat = -(soft * logp).sum(1).mean()
        l_split = F.binary_cross_entropy_with_logits(s.float(), teacher[:, 2], pos_weight=pos_w)
        l_eject = F.binary_cross_entropy_with_logits(e.float(), teacher[:, 3], pos_weight=pos_w)
        loss = l_heat + l_split + l_eject
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        step += 1
        sh.trained += a.batch
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            torch._foreach_mul_(list(ema.parameters()), d)
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - d)
        if step % 100 == 0:
            with torch.no_grad():
                ang, _, _ = target_metrics(decode_target(heat), tgt)
                ok = (ang < 15).float()
                acc.setdefault("heat", []).append(l_heat.item())
                acc.setdefault("split", []).append(l_split.item())
                acc.setdefault("ok", []).append(ok.mean().item())
            sh.infer_sd = {k: v.clone() for k, v in ema.state_dict().items()}
            sh.infer_ver += 1
        if step % 500 == 0:
            now = time.time()
            rec = dict(t=round(el), step=step, samples=sh.trained, lr=lr, p_student=sh.p_student,
                       heat=np.mean(acc["heat"]), split=np.mean(acc["split"]), train_agree=np.mean(acc["ok"]),
                       gnorm=gn.item(), train_rate=500 * a.batch / (now - t_log), env_rate=sh.env_rate,
                       deaths=sh.deaths, pos_events=pos.t, danger=sh.danger_flagged)
            acc.clear()
            t_log = now
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            logf.write(json.dumps(dict(kind="train", **rec)) + "\n")
            logf.flush()
        if time.time() >= next_eval:
            res = evaluate(ema, val_main, val_ev)
            sh.split_thr = float(np.clip(res["split_best"]["thr"], 0.05, 0.9))
            res.update(kind="val", t=round(time.time() - t_start), step=step, samples=sh.trained, p_student=sh.p_student)
            print("VAL", json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), "split",
                  json.dumps(res["split_best"]), flush=True)
            logf.write(json.dumps(res) + "\n")
            logf.flush()
            save("last", res)
            save(f"ck_{int(time.time() - t_start):05d}", res)
            if res["target_agree15"] > best:
                best = res["target_agree15"]
                save("best", res)
            next_eval = time.time() + a.eval_min * 60
    sh.stop = True
    th.join(timeout=30)
    res = evaluate(ema, val_main, val_ev)
    res.update(kind="final", t=round(time.time() - t_start), step=step, samples=sh.trained)
    logf.write(json.dumps(res) + "\n")
    print("FINAL", json.dumps(res), flush=True)
    save("last", res)
    if res["target_agree15"] > best:
        save("best", res)
    if sh.error:
        raise sh.error


if __name__ == "__main__":
    main()
