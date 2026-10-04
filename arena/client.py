"""Headless MultiOgar client (protocol 6) that plays one agent in lockstep.

Keeps a cell table from UpdateNodes deltas, and on every 0xC9 step packet asks
the policy for an action, sends it, then acknowledges the step.
"""

import struct
from dataclasses import dataclass, field

import websockets

from bot.game import Cell, Snapshot

OP_AGENT, OP_STEP = 0xC8, 0xC9


def _read_str(buf: bytes, i: int) -> tuple[str, int]:
    end = buf.index(0, i)
    return buf[i:end].decode("utf-8", "replace"), end + 1


def _kind(flags: int) -> str:
    if flags & 0x80:
        return "food"
    if flags & 0x11:
        return "virus"
    if flags & 0x20:
        return "ejected"
    return "player"


@dataclass
class AgentStats:
    name: str
    steps_alive: int = 0
    mass_ticks: float = 0.0   # integral of mass over game ticks
    peak_mass: float = 0.0
    deaths: int = 0
    kills: int = 0
    lives: int = 0
    ticks: int = 0
    counters: dict = field(default_factory=dict)  # policy skill usage (ejects, shots, ...)

    @property
    def avg_mass(self) -> float:
        return self.mass_ticks / max(self.ticks, 1)


@dataclass
class _State:
    cells: dict = field(default_factory=dict)   # id -> Cell (kind "player" etc.)
    mine: set = field(default_factory=set)
    border: dict = field(default_factory=lambda: {"left": -1, "right": 1, "top": -1, "bottom": 1})

    def snapshot(self) -> Snapshot:
        mine, others = [], []
        for c in self.cells.values():
            (mine if c.id in self.mine else others).append(c)
        for c in mine:
            c.kind = "mine"
        return Snapshot(mine, others, {"x": 0, "y": 0, "z": 1}, 0, 0, self.border, not mine)


class AgentClient:
    def __init__(self, url: str, name: str, policy, end_tick: int, sync: int):
        self.url = url
        self.name = name
        self.policy = policy
        self.end_tick = end_tick
        self.sync = sync
        self.stats = AgentStats(name)
        self._st = _State()
        self.virus = (0, 0, 0.0)  # cumulative (virus shots, enemies popped, popped mass)

    def _on_update(self, buf: bytes) -> None:
        st = self._st
        i = 1
        (n,) = struct.unpack_from("<H", buf, i)
        i += 2
        for _ in range(n):
            killer, killed = struct.unpack_from("<II", buf, i)
            i += 8
            victim = st.cells.get(killed)
            if (killer in st.mine and victim is not None and victim.kind == "player"
                    and killed not in st.mine):
                self.stats.kills += 1
            st.cells.pop(killed, None)
            st.mine.discard(killed)
        while True:
            (cid,) = struct.unpack_from("<I", buf, i)
            i += 4
            if cid == 0:
                break
            x, y, size, flags = struct.unpack_from("<iiHB", buf, i)
            i += 11
            if flags & 0x02:
                i += 3
            if flags & 0x04:
                _, i = _read_str(buf, i)
            name = None
            if flags & 0x08:
                name, i = _read_str(buf, i)
            c = st.cells.get(cid)
            if c is None:
                st.cells[cid] = Cell(cid, x, y, size, _kind(flags), name or "")
            else:
                c.x, c.y, c.size = x, y, size
                if name:
                    c.name = name
        (n,) = struct.unpack_from("<H", buf, i)
        i += 2
        for _ in range(n):
            (cid,) = struct.unpack_from("<I", buf, i)
            i += 4
            st.cells.pop(cid, None)
            st.mine.discard(cid)

    def _on_step(self, tick: int) -> list[bytes]:
        """Advance stats and return the packets to send for this step."""
        s = self._st.snapshot()
        out = []
        if s.dead:
            if self.stats.steps_alive and self._alive:
                self.stats.deaths += 1
            self._alive = False
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            out.append(b"\x00" + self.name.encode() + b"\x00")
        else:
            if not self._alive:
                self.stats.lives += 1
            self._alive = True
            mass = s.total_mass
            self.stats.steps_alive += 1
            self.stats.mass_ticks += mass * self.sync
            self.stats.peak_mass = max(self.stats.peak_mass, mass)
            x, y, split, *rest = self.policy.act(s)
            out.append(struct.pack("<BiiI", 0x10, int(x), int(y), 0))
            if split:
                out.append(b"\x11")
            if rest and rest[0]:
                out.append(b"\x15")  # W: eject mass toward the mouse
        self.stats.ticks += self.sync
        out.append(struct.pack("<BI", OP_STEP, tick))
        return out

    def _handle(self, msg: bytes) -> int | None:
        """Apply one server packet to the cell table; returns the tick for step packets."""
        op = msg[0]
        if op == 0x10:
            self._on_update(msg)
        elif op == 0x20:
            self._st.mine.add(struct.unpack_from("<I", msg, 1)[0])
        elif op == 0x12:
            self._st.cells.clear()
            self._st.mine.clear()
        elif op == 0x14:
            self._st.mine.clear()
        elif op == 0x40:
            left, top, right, bottom = struct.unpack_from("<dddd", msg, 1)
            self._st.border = {"left": left, "top": top, "right": right, "bottom": bottom}
        elif op == OP_STEP:
            if len(msg) >= 17:  # cumulative virus shots / pops / popped mass (arena.js)
                self.virus = struct.unpack_from("<IIf", msg, 5)
            return struct.unpack_from("<I", msg, 1)[0]
        return None

    async def run(self) -> AgentStats:
        self._alive = False
        async with websockets.connect(self.url, max_size=None, compression=None,
                                      ping_interval=None) as ws:
            await ws.send(bytes([254, 6, 0, 0, 0]))
            await ws.send(bytes([255, 1, 0, 0, 0]))
            await ws.send(bytes([OP_AGENT]))
            started_at = None
            async for msg in ws:
                tick = self._handle(msg)
                if tick is not None:
                    started_at = tick if started_at is None else started_at
                    for packet in self._on_step(tick):
                        await ws.send(packet)
                    if tick - started_at >= self.end_tick:
                        break
        self.stats.counters = dict(getattr(self.policy, "counters", {}))
        return self.stats


class StepConn(AgentClient):
    """Externally driven agent: `await next_step()`, read `snapshot()`, then `await act(...)`.

    Used by RL environments, where a learner (not a local policy) chooses actions.
    """

    def __init__(self, url: str, name: str, sync: int):
        super().__init__(url, name, policy=None, end_tick=0, sync=sync)
        self.ws = None
        self.tick = 0

    async def open(self) -> None:
        self.ws = await websockets.connect(self.url, max_size=None, compression=None,
                                           ping_interval=None)
        for packet in (bytes([254, 6, 0, 0, 0]), bytes([255, 1, 0, 0, 0]), bytes([OP_AGENT])):
            await self.ws.send(packet)

    async def next_step(self) -> int:
        while True:
            tick = self._handle(await self.ws.recv())
            if tick is not None:
                self.tick = tick
                return tick

    def snapshot(self) -> Snapshot:
        return self._st.snapshot()

    async def act(self, x: float, y: float, split: bool = False, eject: bool = False) -> None:
        """Send this step's input (or a respawn when dead) and acknowledge the step."""
        if self.snapshot().dead:
            await self.ws.send(b"\x00" + self.name.encode() + b"\x00")
        else:
            await self.ws.send(struct.pack("<BiiI", 0x10, int(x), int(y), 0))
            if split:
                await self.ws.send(b"")
            if eject:
                await self.ws.send(b"")
        await self.ws.send(struct.pack("<BI", OP_STEP, self.tick))

    async def set_spawn_size(self, size: int) -> None:
        """Arena-only: cell size for this agent's next spawns (0 = server default)."""
        await self.ws.send(struct.pack("<BH", 0xCB, max(0, min(int(size), 65535))))

    async def ack(self) -> None:
        """Acknowledge the step without new input (the server keeps the last mouse)."""
        await self.ws.send(struct.pack("<BI", OP_STEP, self.tick))

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
