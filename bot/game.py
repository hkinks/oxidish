"""Browser side: read client state through the `__bot` hook, drive real mouse/keys."""

import math
from dataclasses import dataclass

# Runs in the page. Only reads; the hook is added by client-hook.patch.
_SNAPSHOT_JS = """
() => {
  const b = window.__bot;
  if (!b || !b.canvas) return null;
  const mine = new Set(b.cells.mine);
  const out = [];
  for (const c of b.cells.byId.values()) {
    if (c.destroyed) continue;
    out.push([c.id, c.x, c.y, c.s,
      c.food ? 'food' : c.jagged ? 'virus' : c.ejected ? 'ejected' : mine.has(c.id) ? 'mine' : 'player',
      c.name || '']);
  }
  return {
    cells: out,
    camera: {x: b.camera.x, y: b.camera.y, z: b.camera.z},
    width: b.canvas.width, height: b.canvas.height,
    border: {left: b.border.left, right: b.border.right, top: b.border.top, bottom: b.border.bottom},
    dead: b.overlayShown || b.cells.mine.length === 0,
  };
}
"""


@dataclass
class Cell:
    id: int
    x: float
    y: float
    size: float  # radius in world units
    kind: str    # food | virus | ejected | mine | player
    name: str
    vx: float = 0.0  # per-step velocity, filled in by policies that track cells
    vy: float = 0.0

    @property
    def mass(self) -> float:
        return self.size * self.size / 100

    def dist(self, other: "Cell") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Snapshot:
    mine: list[Cell]
    others: list[Cell]
    camera: dict
    width: int
    height: int
    border: dict
    dead: bool

    @property
    def main(self) -> Cell | None:
        return max(self.mine, key=lambda c: c.size, default=None)

    @property
    def total_mass(self) -> float:
        return sum(c.mass for c in self.mine)


class Game:
    def __init__(self, page, name: str):
        self._page = page
        self._name = name

    def snapshot(self) -> Snapshot | None:
        raw = self._page.evaluate(_SNAPSHOT_JS)
        if raw is None:
            return None
        cells = [Cell(*c) for c in raw["cells"]]
        return Snapshot(
            mine=[c for c in cells if c.kind == "mine"],
            others=[c for c in cells if c.kind != "mine"],
            camera=raw["camera"], width=raw["width"], height=raw["height"],
            border=raw["border"], dead=raw["dead"],
        )

    def respawn(self) -> None:
        # Same as typing a nickname and pressing Play on the overlay.
        self._page.evaluate("name => play(name)", self._name)

    def aim(self, s: Snapshot, wx: float, wy: float) -> None:
        """Point the real mouse at world coordinate (wx, wy)."""
        cam = s.camera
        sx = (wx - cam["x"]) * cam["z"] + s.width / 2
        sy = (wy - cam["y"]) * cam["z"] + s.height / 2
        sx = min(max(sx, 1), s.width - 2)
        sy = min(max(sy, 1), s.height - 2)
        self._page.mouse.move(sx, sy)

    def split(self) -> None:
        self._page.keyboard.press("Space")
