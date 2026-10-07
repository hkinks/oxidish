"""Shared helpers for the M2 imitation scripts: process affinity, the opponent field, metrics."""
import ctypes
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The endless-eval field: champ, r2g9, safe, ramp_r2, ramp300 (one each) + 8 server bots (VecEnv default `bots=8`).
FIELD = [
    ("champ", "runs/final_champion.json", 1),
    ("r2g9", "runs/r2/champion_009.json", 1),
    ("safe", "bot/champ_safe.json", 1),
    ("ramp_r2", "bot/r2g9_ramp800.json", 1),
    ("ramp300", "runs/r2g9_ramp300.json", 1),
]
TEACHER = "bot/plan_c34.json"
VAL_DIR = "runs/vision/val"


def pin_cores(n_cores=11):
    """Restrict this process (and the env worker threads it spawns) to cores 0..n_cores-1 (live servers use 11-15)."""
    if os.name == "nt":
        k = ctypes.windll.kernel32
        k.SetProcessAffinityMask(k.GetCurrentProcess(), ctypes.c_size_t((1 << n_cores) - 1))


def angle_deg(a, b):
    """Angle between 2-vectors a, b [B,2] in degrees (0 if either is ~zero)."""
    na, nb = a.norm(dim=1), b.norm(dim=1)
    cos = (a * b).sum(1) / (na * nb).clamp_min(1e-6)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def target_metrics(pred_uv, teacher_uv, px=256):
    """Heading agreement from the frame centre + pixel distance. Returns tensors per sample."""
    c = px / 2
    ctr = torch.tensor([c, c], device=pred_uv.device)
    ang = angle_deg(pred_uv - ctr, teacher_uv - ctr)
    dist = (pred_uv - teacher_uv).norm(dim=1)
    tlen = (teacher_uv - ctr).norm(dim=1)
    return ang, dist, tlen


def binary_report(prob, label, thr):
    pred = prob > thr
    tp = (pred & label).sum().item()
    fp = (pred & ~label).sum().item()
    fn = (~pred & label).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(thr=float(thr), precision=prec, recall=rec, f1=f1, acc=1 - (fp + fn) / max(len(label), 1),
                n_pos=int(label.sum().item()), n_pred=int(pred.sum().item()))


def best_threshold(prob, label):
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        r = binary_report(prob, label, thr)
        if best is None or r["f1"] > best["f1"]:
            best = r
    return best
