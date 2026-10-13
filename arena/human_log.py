"""Read `ogar serve --record DIR` recordings (events / scenes / humans, gzip JSONL) and classify loss scenes.

Used by arena.human_report. Truncated gzip files (a killed server) are read as far
as they go.
"""

import glob
import json
import math
import re
import zlib
from pathlib import Path

EAT_RATIO = 1.3          # mass ratio needed to eat a cell (MultiOgar game rule)


KINDS = {0: "player", 1: "food", 2: "virus", 3: "ejected"}


def read_jsonl(path) -> list[dict]:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        txt = d.decompress(Path(path).read_bytes())
    except zlib.error:
        txt = b""
    out = []
    for ln in txt.decode("utf-8", "ignore").split("\n"):
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def sessions(rec_dir) -> list[str]:
    """Session stamps present in a recording dir, oldest first."""
    stamps = {m.group(1) for p in glob.glob(str(Path(rec_dir) / "events-*.jsonl.gz"))
              if (m := re.search(r"events-(\d+)\.jsonl\.gz$", p))}
    return sorted(stamps, key=int)


def load(rec_dir, session: str | None = None) -> dict:
    """{'events': [...], 'scenes': [...], 'sessions': [stamps]}; scenes get an 'id' = '<stamp>-<n>'."""
    stamps = [session] if session else sessions(rec_dir)
    events, scenes = [], []
    for st in stamps:
        for e in read_jsonl(Path(rec_dir) / f"events-{st}.jsonl.gz"):
            e["session"] = st
            events.append(e)
        for i, sc in enumerate(read_jsonl(Path(rec_dir) / f"scenes-{st}.jsonl.gz")):
            sc["id"] = f"{st}-{i}"
            sc["session"] = st
            scenes.append(sc)
    return {"events": events, "scenes": scenes, "sessions": stamps}


def family(name: str, kind: str) -> str:
    """Bot family: 'safe-1' -> 'safe'; server bots -> 'server-bot'; humans -> 'human'."""
    if kind == "bot":
        return "server-bot"
    if kind == "human":
        return "human"
    return re.sub(r"-\d+$", "", name)


# ---------------------------------------------------------------- scenes

def _cells(step) -> tuple[list, list]:
    """(players, viruses) of a scene step: [id, x, y, size, kind, owner]."""
    pl = [o for o in step["o"] if o[4] == 0]
    vi = [o for o in step["o"] if o[4] == 2]
    return pl, vi


def _mass(size) -> float:
    return size * size / 100


def wall_dist(step, border) -> float:
    """Distance from our main cell's edge to the nearest wall at this step."""
    m = max(step["mine"], key=lambda c: c[3])
    l, r, t, b = border
    return min(m[1] - l, r - m[1], m[2] - t, b - m[2]) - m[3]


def _owners(players):
    by = {}
    for o in players:
        by.setdefault(o[5], []).append(o)
    return by


def near_players(step, limit=6, max_gap=900):
    """Players near our main cell, nearest gap first, with their owner's piece count and total mass."""
    me = max(step["mine"], key=lambda c: c[3])
    pl, _ = _cells(step)
    by = _owners(pl)
    out = []
    for o in pl:
        gap = math.hypot(o[1] - me[1], o[2] - me[2]) - o[3] - me[3]
        if gap > max_gap:
            continue
        own = by[o[5]]
        out.append({"name": o[5], "m": round(_mass(o[3])), "gap": round(gap),
                    "dx": o[1] - me[1], "dy": o[2] - me[2], "own_n": len(own),
                    "own_m": round(sum(_mass(c[3]) for c in own))})
    out.sort(key=lambda d: d["gap"])
    return out[:limit]


def classify_scene(sc) -> dict:
    """Cause + tags of a loss scene: popped / split_kill / pieces / unseen / fled / no_flee.
    Tags: wall (main cell edge < 150 from a wall), corner, human (a human ate), solo/multi eater."""
    steps = sc["steps"]
    if not steps:
        return {"cause": "unknown", "tags": [], "wall": None}
    look = max(1, round(25 / sc.get("sync", 2)))          # 1 s of steps
    last, early = steps[-1], steps[max(0, len(steps) - 1 - look)]
    counts = [len(s["mine"]) for s in steps]
    popped = any(b >= a + 3 for a, b in zip(counts[-7:], counts[-6:]))
    my_main = _mass(max(c[3] for c in early["mine"]))
    smallest = min(_mass(c[3]) for c in early["mine"])
    eaters_seen = [p for p in near_players(early, 99, 1200) if p["m"] > smallest * EAT_RATIO]
    split_kill = any(e["eater"]["cells"] >= 2 and e["eater"]["tot"] > my_main * EAT_RATIO
                     and e["eater"]["cm"] < my_main * EAT_RATIO for e in sc["eaters"])
    fled = any(s["mode"] == "flee" for s in steps[-look:])
    pieces = len(early["mine"]) >= 2 and sc["kind"] != "death"
    if popped:
        cause = "popped"
    elif split_kill:
        cause = "split_kill"
    elif pieces:
        cause = "pieces"
    elif not eaters_seen:
        cause = "unseen"
    elif fled:
        cause = "fled"
    else:
        cause = "no_flee"
    wd = wall_dist(last, sc["border"])
    l, r, t, b = sc["border"]
    me = max(last["mine"], key=lambda c: c[3])
    near_x = min(me[1] - l, r - me[1]) - me[3] < 400
    near_y = min(me[2] - t, b - me[2]) - me[3] < 400
    tags = []
    if wd < 150:
        tags.append("wall")
    if near_x and near_y:
        tags.append("corner")
    if sc.get("human_eater"):
        tags.append("human")
    return {"cause": cause, "tags": tags, "wall": round(wd), "fled": fled, "popped": popped, "look": look}


def render_scene(sc, width=61, height=25, span=1000) -> str:
    """Compact text rendering: header, a decision table (every ~0.5 s), and an ASCII map of the last step.
    Map legend: @ our cell, E player that can eat our main cell, p player we can eat, o other player,
    V virus, # wall; scale span units from the centre to the edge of the map."""
    steps = sc["steps"]
    cl = classify_scene(sc)
    lines = []
    eat = "; ".join(f"{e['eater']['n']}({e['eater']['k']}) cell {e['eater']['cm']} total {e['eater']['tot']} x{e['eater']['cells']}"
                    f" ate {e['victim']['cm']}{' [popped]' if e['popped'] else ''}" for e in sc["eaters"][:4])
    lines.append(f"scene {sc['id']}  {sc['agent']}  tick {sc['t']}  {sc['kind']} {sc['before']} -> {sc['after']}"
                 f"  cause={cl['cause']} tags={','.join(cl['tags']) or '-'} wall_dist={cl['wall']}")
    lines.append(f"  eaten by: {eat or '?'}")
    stride = max(1, round(12.5 / sc.get("sync", 2)))   # one row per ~0.5 s
    lines.append("  t-s  mode(stage) total cells  aim(dx,dy)    nearest players (mass, gap, bearing dx/dy)")
    last_t = steps[-1]["t"]
    for st in steps[::-1][::stride][::-1]:
        me = max(st["mine"], key=lambda c: c[3])
        tot = round(sum(_mass(c[3]) for c in st["mine"]))
        near = near_players(st, 3)
        ns = " ".join(f"{p['name'][:9]}:{p['m']}m g{p['gap']} ({p['dx']:+d},{p['dy']:+d})x{p['own_n']}" for p in near)
        flag = "SPLIT" if st["split"] else ""
        lines.append(f"  {(st['t'] - last_t) / 25:+5.1f} {st['mode']:>5}({st['stage'][:3]}) {tot:6d} {len(st['mine']):3d}  "
                     f"({st['aim'][0] - me[1]:+5.0f},{st['aim'][1] - me[2]:+5.0f}) {flag:5} {ns}")
    # map
    st = steps[-1]
    me = max(st["mine"], key=lambda c: c[3])
    grid = [[" "] * width for _ in range(height)]
    sx = span / (width / 2)
    sy = 2 * sx          # characters are about twice as tall as wide

    def put(x, y, ch):
        cx, cy = int(round((x - me[1]) / sx + width / 2)), int(round((y - me[2]) / sy + height / 2))
        if 0 <= cx < width and 0 <= cy < height:
            grid[cy][cx] = ch

    l, r, t, b = sc["border"]
    for gy in range(height):
        for gx in range(width):
            wx, wy = me[1] + (gx - width / 2) * sx, me[2] + (gy - height / 2) * sy
            if wx < l or wx > r or wy < t or wy > b:
                grid[gy][gx] = "#"
    my_m, small = _mass(me[3]), min(_mass(c[3]) for c in st["mine"])
    for o in st["o"]:
        if o[4] == 2:
            put(o[1], o[2], "V")
        elif o[4] == 0:
            m = _mass(o[3])
            put(o[1], o[2], "E" if m > small * EAT_RATIO else "p" if my_m > m * EAT_RATIO else "o")
    for c in st["mine"]:
        put(c[1], c[2], "@")
    lines.append(f"  map at loss (N up, +-{span} units, one char = {sx:.0f} x {sy:.0f}):")
    lines += ["  |" + "".join(row) + "|" for row in grid]
    return "\n".join(lines)
