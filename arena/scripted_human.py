"""A scripted "human" for testing the recording / human-report pipeline without a person.

    uv run python -m arena.scripted_human --port 9650 --seconds 120 --boost 3

Connects like the native client (handshake 254/6, 255/1, spawn 0 + name, mouse 0x10 x y int32 + 4 bytes,
split 17, +500 mass 0xCD), then hunts: chases the best-value smaller player cell in view, runs from bigger
ones, splits when a split would eat the target. It is deliberately aggressive, so it eats bots and the
server's loss scenes have plenty of material.
"""

import argparse
import asyncio
import math
import struct
import time

import websockets

from bot.steer import EAT_RATIO

from .client import AgentClient

STEP = 0.08  # seconds between decisions (the native client sends the mouse ~every frame)


def decide(s):
    """-> (x, y, split). Greedy: nearest worthwhile prey, flee from anything that can eat our main cell."""
    me = s.main
    mine = {c.id for c in s.mine}
    best, best_k = None, 0.0
    flee = None
    for c in s.others:
        if c.kind != "player" or c.id in mine:
            continue
        gap = me.dist(c) - c.size - me.size
        if c.mass > me.mass * EAT_RATIO and gap < 350 + me.size:
            if flee is None or gap < flee[0]:
                flee = (gap, c)
        elif me.mass > c.mass * EAT_RATIO:
            k = c.mass / (max(gap, 0) + 100)
            if k > best_k:
                best, best_k = c, k
    if flee is not None:
        c = flee[1]
        return me.x + (me.x - c.x) * 3, me.y + (me.y - c.y) * 3, False
    if best is not None:
        gap = me.dist(best) - best.size - me.size
        split = len(s.mine) < 3 and me.mass / 2 > best.mass * EAT_RATIO and 0 < gap < 220 and me.mass > 60
        return best.x, best.y, split
    # nothing to hunt: drift to the middle of the map
    b = s.border
    return (b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2, False


async def run(url: str, name: str, seconds: float, boost: int, rejoin_boost: bool) -> None:
    cl = AgentClient(url, name, None, 0, 1)
    end = time.time() + seconds
    async with websockets.connect(url, max_size=None, compression=None, ping_interval=None) as ws:
        await ws.send(bytes([254, 6, 0, 0, 0]))
        await ws.send(bytes([255, 1, 0, 0, 0]))

        async def reader():
            async for msg in ws:
                if isinstance(msg, bytes) and msg:
                    cl._handle(msg)

        task = asyncio.create_task(reader())
        was_dead, born = True, 0.0
        try:
            while time.time() < end:
                s = cl._st.snapshot()
                if s.dead:
                    if not was_dead or time.time() - born > 1.0:
                        await ws.send(b"\x00" + name.encode() + b"\x00")
                        born = time.time()
                    was_dead = True
                else:
                    if was_dead:
                        was_dead = False
                        if boost and (rejoin_boost or not hasattr(cl, "_boosted")):
                            cl._boosted = True
                            for _ in range(boost):
                                await ws.send(b"\xcd")
                    x, y, split = decide(s)
                    await ws.send(struct.pack("<BiiI", 0x10, int(x), int(y), 0))
                    if split:
                        await ws.send(b"\x11")
                await asyncio.sleep(STEP)
        finally:
            task.cancel()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--name", default="human")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--boost", type=int, default=3, help="number of +500 mass presses (0xCD) after each spawn")
    p.add_argument("--first-boost-only", action="store_true", help="boost only the first life")
    a = p.parse_args()
    asyncio.run(run(f"ws://{a.host}:{a.port}", a.name, a.seconds, a.boost, not a.first_boost_only))


if __name__ == "__main__":
    main()
