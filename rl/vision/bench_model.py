"""Training / inference speed of VisionNet at a few widths:  python rl/vision/bench_model.py"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rl.vision.model import VisionNet  # noqa: E402

for width in (0.5, 1.0, 1.5):
    m = VisionNet(width).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    n = sum(p.numel() for p in m.parameters())
    B = 256
    v = torch.randint(0, 255, (B, 2, 256, 256), dtype=torch.uint8, device="cuda")
    w = torch.randint(0, 255, (B, 2, 128, 128), dtype=torch.uint8, device="cuda")
    for i in range(13):
        if i == 3:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast("cuda", torch.bfloat16):
            h, s, e, val = m(v, w)
            loss = h.float().logsumexp((1, 2)).mean() + s.float().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / 10
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        v1, w1 = v[:64], w[:64]
        for i in range(13):
            if i == 3:
                torch.cuda.synchronize()
                t = time.time()
            m(v1, w1)
        torch.cuda.synchronize()
        di = (time.time() - t) / 10
    print(f"width {width}: {n / 1e6:.2f}M params, train {B / dt:.0f} samples/s, infer b64 {di * 1000:.1f} ms, "
          f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
