"""Parity of the Rust vision agent (ONNX Runtime) against torch on recorded frames.

    ogar vdump --agents vis=bot/vision_bc3.json:2 champ=runs/final_champion.json:2 --count 1000 --out logs/vision/parity
    python rl/vision/parity.py logs/vision/parity runs/vision/export/vision_bc3.pt

Loads view.u8 / wide.u8 / rust.f32 written by `ogar vdump`, runs the TorchScript export on the same frames (fp32,
CPU), decodes the target with model.decode_target and compares with the Rust target: pixel distance and the angle
between the two headings seen from the frame centre (what the game uses), plus the split-logit difference.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("rl", 1)[0] + "rl/vision")
from model import decode_target  # noqa: E402


def main(d, ts_path, device="cpu"):
    rust = np.fromfile(f"{d}/rust.f32", dtype="<f4").reshape(-1, 4)
    n = len(rust)
    view = torch.from_numpy(np.fromfile(f"{d}/view.u8", dtype=np.uint8).reshape(n, 2, 256, 256))
    wide = torch.from_numpy(np.fromfile(f"{d}/wide.u8", dtype=np.uint8).reshape(n, 2, 128, 128))
    model = torch.jit.load(ts_path, map_location=device).eval()
    tg, sp = [], []
    with torch.no_grad():
        for i in range(0, n, 50):
            heat, split, eject, value = model(view[i:i + 50].to(device), wide[i:i + 50].to(device))
            tg.append(decode_target(heat).cpu())
            sp.append(split.cpu())
    tg, sp = torch.cat(tg).numpy(), torch.cat(sp).numpy()
    px = np.hypot(*(tg - rust[:, :2]).T)
    c = 128.0

    def ang(t):
        return np.degrees(np.arctan2(t[:, 1] - c, t[:, 0] - c))

    da = np.abs((ang(tg) - ang(rust[:, :2]) + 180) % 360 - 180)
    ds = np.abs(sp - rust[:, 2])
    print(f"{n} frames  (torch {ts_path} on {device})")
    print(f"target pixel distance: median {np.median(px):.3f}  p99 {np.percentile(px, 99):.2f}  max {px.max():.2f}")
    print(f"heading difference (deg): median {np.median(da):.3f}  p99 {np.percentile(da, 99):.2f}  max {da.max():.2f}")
    print(f"frames within 1 deg: {(da < 1).mean():.1%}   within 0.1 deg: {(da < 0.1).mean():.1%}   same cell: {(px < 4).mean():.1%}")
    print(f"split logit abs diff: median {np.median(ds):.4f}  max {ds.max():.3f}")
    print(f"split decision (sigmoid > 0.85) differs in {((1 / (1 + np.exp(-sp)) > 0.85) != (1 / (1 + np.exp(-rust[:, 2])) > 0.85)).sum()} frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cpu")
