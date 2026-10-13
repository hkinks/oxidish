"""Held-out validation metrics (target agreement, split / eject precision-recall) for a VisionNet.

    python training/rl/evalval.py runs/vision/<name>/best.pt
"""
import glob
import os
import sys

import numpy as np
import torch

from common import VAL_DIR, ROOT, best_threshold, binary_report, target_metrics  # noqa: F401
from model import VisionNet, decode_target
from store import FrameStore


def load_windows(pattern, device="cuda"):
    """Concatenate the saved windows along the learner axis into one FrameStore (rows 1.. are evaluated)."""
    files = sorted(glob.glob(pattern))
    parts = [np.load(f) for f in files]
    T = parts[0]["view"].shape[0]
    N = sum(p["view"].shape[1] for p in parts)
    st = FrameStore(T, N, device=device)
    col = 0
    for p in parts:
        n = p["view"].shape[1]
        st.view[:, col:col + n] = torch.from_numpy(p["view"]).to(device)
        st.wide[:, col:col + n] = torch.from_numpy(p["wide"]).to(device)
        st.teacher[:, col:col + n] = torch.from_numpy(p["teacher"]).to(device)
        st.first[:, col:col + n] = torch.from_numpy(p["first"]).to(device).bool()
        col += n
    st.filled = T
    st.t = T
    return st


@torch.no_grad()
def predict(model, st, bs=256):
    """Run the model on every evaluated frame of a window store. Returns dict of CPU/GPU tensors, flattened."""
    model.eval()
    out = dict(uv=[], split=[], eject=[], tgt=[], tsplit=[], teject=[], ce=[])
    for r in range(1, st.T):
        for c0 in range(0, st.N, bs):
            cols = torch.arange(c0, min(c0 + bs, st.N), device=st.dev)
            rows = torch.full_like(cols, r)
            view, wide, teacher, _ = st.gather(rows, cols)
            with torch.autocast("cuda", torch.bfloat16):
                heat, s, e, _ = model(view, wide)
            out["uv"].append(decode_target(heat.float()))
            out["split"].append(torch.sigmoid(s.float()))
            out["eject"].append(torch.sigmoid(e.float()))
            out["tgt"].append(teacher[:, :2])
            out["tsplit"].append(teacher[:, 2] > 0.5)
            out["teject"].append(teacher[:, 3] > 0.5)
    model.train()
    return {k: torch.cat(v) for k, v in out.items() if v}


def summarize(pr, far_px=24.0):
    ang, dist, tlen = target_metrics(pr["uv"], pr["tgt"])
    far = tlen > far_px
    res = dict(
        n=int(len(ang)),
        target_agree15=(ang < 15).float().mean().item(),
        target_agree15_far=(ang[far] < 15).float().mean().item() if far.any() else float("nan"),
        target_agree30=(ang < 30).float().mean().item(),
        median_angle=ang.median().item(), median_px=dist.median().item(), mean_px=dist.mean().item(),
        far_frac=far.float().mean().item(),
    )
    res["split_at0.5"] = binary_report(pr["split"], pr["tsplit"], 0.5)
    res["split_best"] = best_threshold(pr["split"], pr["tsplit"])
    if pr["teject"].any():
        res["eject_at0.5"] = binary_report(pr["eject"], pr["teject"], 0.5)
    return res


def evaluate(model, val_main, val_events=None):
    res = summarize(predict(model, val_main))
    if val_events is not None:
        pe = predict(model, val_events)
        res["event_split_recall_at0.5"] = (pe["split"] > 0.5).float().mean().item()
        thr = res["split_best"]["thr"]
        res["event_split_recall_at_best"] = (pe["split"] > thr).float().mean().item()
        res["event_target_agree15"] = (target_metrics(pe["uv"], pe["tgt"])[0] < 15).float().mean().item()
    return res


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = VisionNet(**ck["cfg"]).to(device).to(memory_format=torch.channels_last)
    m.load_state_dict(ck["model"])
    return m.eval(), ck


if __name__ == "__main__":
    os.chdir(ROOT)
    m, ck = load_model(sys.argv[1])
    vm = load_windows(os.path.join(VAL_DIR, "win_*.npz"))
    ve = load_windows("runs/vision/val_events/events.npz") if os.path.exists("runs/vision/val_events/events.npz") else None
    import json
    print(json.dumps(evaluate(m, vm, ve), indent=1))
