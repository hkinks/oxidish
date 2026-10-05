"""Human play recorded on the showcase (arena/server/arena.js --record) -> imitation data for entity nets.

    uv run python -m rl.human_data [--names human] [--out runs/human/dataset.pt]

Every recorded frame of a human player becomes one sample: the entity tokens and scalars an entity
net would see (rl/entity_obs.RealEntityEncoder, fed the frames in order so velocities, merge timers
and the commitment memory build up as in play) and the human's action as the label:
  move   - the heading (of 16) from the view centre to the mouse target; "stay" (16) when the mouse
           is within 30 units of the centre;
  split  - 1 if split was pressed since the last frame; eject - 1 if W was pressed.
A life ends when the player has no cells or a frame is missing (death / respawn): the encoder resets.
Recordings written by a killed server can end mid-block; the readable part is used.
"""

import argparse
import glob
import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch

from bot.game import Cell, Snapshot
from rl.common import ROOT

from .entity_obs import RealEntityEncoder

KIND = {0: "player", 1: "food", 2: "virus", 3: "ejected"}
DIRS = 16


def read_frames(path: Path) -> list[dict]:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        txt = d.decompress(path.read_bytes())
    except zlib.error:
        txt = b""
    out = []
    for line in txt.decode("utf-8", "ignore").split("\n"):
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            pass
    return out


def snapshot(fr: dict) -> Snapshot:
    mine = [Cell(i, x, y, s, "mine", fr["n"]) for i, x, y, s in fr["mine"]]
    others = [Cell(i, x, y, s, KIND.get(t, "food"), owner) for i, x, y, s, t, owner in fr["o"]]
    l, r, t, b = fr["b"]
    return Snapshot(mine, others, {}, 1920, 1080, {"left": l, "right": r, "top": t, "bottom": b}, not mine)


def convert(frames: list[dict], names: set[str]):
    ents, scals, acts = [], [], []
    by_name: dict[str, list[dict]] = {}
    for fr in frames:
        if fr["n"] in names:
            by_name.setdefault(fr["n"], []).append(fr)
    for name, frs in by_name.items():
        frs.sort(key=lambda f: f["t"])
        enc, last_t = RealEntityEncoder(), None
        for fr in frs:
            if not fr["mine"] or (last_t is not None and fr["t"] - last_t > 4):
                enc.reset()                       # death / respawn / gap: a new life
            last_t = fr["t"]
            if not fr["mine"]:
                continue
            ent, scal, cx, cy, ext = enc.encode(snapshot(fr))
            mx, my = fr["m"]
            dx, dy = mx - cx, my - cy
            move = DIRS if math.hypot(dx, dy) < 30 else round(math.atan2(dy, dx) / (2 * math.pi) * DIRS) % DIRS
            split, eject = int(fr["s"] > 0), int(fr["w"] > 0)
            enc.decode(move, split, eject, cx, cy, ext)     # the commitment memory follows the human
            ents.append(ent.astype(np.float16))
            scals.append(scal)
            acts.append((move, split, eject))
    return (torch.from_numpy(np.stack(ents)), torch.from_numpy(np.stack(scals)),
            torch.tensor(acts, dtype=torch.long))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default=str(ROOT / "runs" / "human" / "humans-*.jsonl.gz"))
    p.add_argument("--names", nargs="+", default=["human"], help="player names that are humans")
    p.add_argument("--out", default=str(ROOT / "runs" / "human" / "dataset.pt"))
    args = p.parse_args()
    frames = [f for path in sorted(glob.glob(args.glob)) for f in read_frames(Path(path))]
    ent, scal, act = convert(frames, set(args.names))
    torch.save({"ent": ent, "scal": scal, "act": act}, args.out)
    n = len(act)
    print(f"{n} samples ({n * 0.08 / 60:.1f} min of play) -> {args.out}: splits {int(act[:, 1].sum())}, "
          f"W {int(act[:, 2].sum())}, stay {(act[:, 0] == DIRS).float().mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
