"""Change food density on a running arena server, live.

    uv run python -m arena.food 3            # 3x the default density on port 9500
    uv run python -m arena.food 1 --port 9500

Sends the arena-only 0xCC packet; the server tops food up toward the new target
every tick (lowering it just stops spawning until food is eaten down).
"""

import argparse
import asyncio
import struct

import websockets


async def send(port: int, mult: float) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        await ws.send(struct.pack("<BH", 0xCC, round(mult * 100)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mult", type=float, help="multiplier of the default food density (e.g. 2, 3.5)")
    p.add_argument("--port", type=int, default=9500)
    args = p.parse_args()
    asyncio.run(send(args.port, args.mult))
    print(f"food density set to {args.mult}x on port {args.port}")


if __name__ == "__main__":
    main()
