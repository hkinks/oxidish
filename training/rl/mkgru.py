"""Make a recurrent checkpoint (M5) from a trained one: same weights + a fresh GRU whose projections are zero, so it plays
exactly like the source net until trained.

    python training/rl/mkgru.py runs/vision/p6/final.pt runs/vision/p6g/init.pt [--gru 256]
"""
import os
import sys

import torch

from evalval import load_model
from model import VisionNet


def main(src, dst, gru=256):
    m, ck = load_model(src, device="cpu")
    cfg = dict(ck["cfg"], gru=gru)
    r = VisionNet(**cfg)
    missing = r.load_state_dict(ck["model"], strict=False)
    print("new keys:", missing.missing_keys)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(dict(cfg=cfg, model=r.state_dict(), split_thr=ck.get("split_thr", 0.5), args={}, iters=0), dst)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 256)
