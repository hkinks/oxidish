"""Per-tick steering by potential field. Pure arithmetic; the mode comes from Jev
(browser bot) or bot/modes.py (arena). All tunables live in bot/params.py."""

import math
from dataclasses import dataclass

from .farm import FoodRouter
from .game import Cell, Snapshot
from .params import DEFAULT, Params

EAT_RATIO = 1.3          # mass ratio needed to eat a cell (MultiOgar game rule)
SPLIT_KILL_RATIO = 2.6   # a cell this much heavier can split onto us
SPLIT_RANGE = DEFAULT.split_range  # kept for the Jev strategist's state description
AIM_DISTANCE = 600       # how far ahead of the cell the mouse is placed


@dataclass
class Plan:
    mode: str = "farm"
    target_id: int | None = None
    split_ok: bool = False


def classify(s: Snapshot, p: Params | None = None) -> tuple[list[Cell], list[Cell], list[Cell], list[Cell]]:
    """Split other cells into (food, prey, threats, viruses) relative to our main cell. With `p.contender`,
    a near-equal player within `p.contender_gap` of us is a threat too (one meal makes it one)."""
    me = s.main
    food, prey, threats, viruses = [], [], [], []
    for c in s.others:
        if c.kind in ("food", "ejected"):
            if c.kind == "food" or c.mass * EAT_RATIO < me.mass:
                food.append(c)
        elif c.kind == "virus":
            viruses.append(c)
        elif me.mass > c.mass * EAT_RATIO:
            prey.append(c)
        elif c.mass > me.mass * EAT_RATIO:
            threats.append(c)
        elif (p is not None and p.contender > 0 and c.mass > me.mass * p.contender
              and me.dist(c) - c.size - me.size < p.contender_gap):
            threats.append(c)
    return food, prey, threats, viruses


def half_of(me: Cell) -> Cell:
    """One of the two cells `me` would split into (half the mass, same place)."""
    return Cell(me.id, me.x, me.y, me.size / math.sqrt(2), me.kind, me.name)


def split_threats(s: Snapshot, me: Cell, p: Params) -> list[Cell]:
    """Threats that count for a split decision: to `me`, or (half_split_safety) to the halves after it."""
    if p.half_split_safety <= 0.5:
        return classify(s)[2]
    half, mine = half_of(me), {c.id for c in s.mine}
    return [c for c in s.others if c.kind == "player" and c.id not in mine and c.mass > half.mass * EAT_RATIO]


WALL_ZONE = 700.0        # wall_danger fades out this far from a wall


def wall_extra(me: Cell, border: dict, p: Params) -> float:
    """Extra danger gap when pressed on a wall (wall_danger): there is no room to run, so react earlier.
    Full at the wall, fading linearly to 0 at WALL_ZONE from it."""
    if p.wall_danger <= 0:
        return 0.0
    wd = min(me.x - border["left"], border["right"] - me.x, me.y - border["top"], border["bottom"] - me.y) - me.size
    return p.wall_danger * min(1.0, max(0.0, 1.0 - wd / WALL_ZONE))


def danger_radius(threat: Cell, me: Cell, p: Params = DEFAULT, border: dict | None = None) -> float:
    reach = threat.size + me.size
    if threat.mass > me.mass * SPLIT_KILL_RATIO:
        reach += p.split_range
    if border is not None:
        reach += wall_extra(me, border, p)
    return reach + p.danger_margin


FLEE_HEADINGS = 32
FLEE_CHECKS = (6, 12, 18)    # look-ahead points, in steps of 2 ticks (18 steps = 1.44 s)


def tick_speed(size: float) -> float:
    """Distance a cell moves per server tick (PlayerCell.getSpeed, full mouse distance)."""
    return 88.0 * max(size, 1.0) ** -0.439


PIECE_SHARE = 0.15           # flee_pieces: pieces lighter than this share of our mass are not protected


def guarded_cells(s: Snapshot, p: Params) -> list[Cell]:
    """Our cells the flee protects: the main cell, plus (flee_pieces) every piece >= PIECE_SHARE of our mass."""
    if p.flee_pieces <= 0.5:
        return [s.main]
    total = s.total_mass or 1.0
    return [c for c in s.mine if c is s.main or c.mass >= PIECE_SHARE * total]


def eaters_of(s: Snapshot, q: Cell) -> list[Cell]:
    mine = {c.id for c in s.mine}
    return [c for c in s.others if c.kind == "player" and c.id not in mine and c.mass > q.mass * EAT_RATIO]


def flee_heading(s: Snapshot, me: Cell, threats: list[Cell], poppers: list[Cell], p: Params,
                 pref: tuple[float, float]) -> tuple[float, float]:
    """Heading that keeps us out of every threat's reach longest over the next ~1.5 s.

    Our path is clamped where the server holds a cell (centre >= size/2 from a wall); each threat
    closes in at its own speed; reach = its eating range (centre within size - our size / 3), plus
    its split lunge when it could split onto us. A heading's score is its worst margin over the
    look-ahead points; paths through a virus we'd pop on are ruled out. Ties go to `pref` (the
    potential field's heading)."""
    b = s.border
    guards = []    # per protected cell: position, wall clamp box, speed, its threats
    for q in guarded_cells(s, p):
        qt = threats if q is me else eaters_of(s, q)
        if not qt:
            continue
        guards.append((q, (b["left"] + q.size / 2, b["right"] - q.size / 2, b["top"] + q.size / 2,
                           b["bottom"] - q.size / 2), 2 * tick_speed(q.size),
                       [(t.x, t.y, 2 * tick_speed(t.size),
                         t.size - q.size / 3 + (p.split_range if t.mass > q.mass * SPLIT_KILL_RATIO else 0.0))
                        for t in qt]))
    best, best_score = pref, -math.inf
    for k in range(FLEE_HEADINGS):
        a = 2 * math.pi * k / FLEE_HEADINGS
        ux, uy = math.cos(a), math.sin(a)
        score = math.inf
        for q, (lo_x, hi_x, lo_y, hi_y), v_q, th in guards:
            for n in FLEE_CHECKS:
                x = min(max(q.x + ux * v_q * n, lo_x), hi_x)
                y = min(max(q.y + uy * v_q * n, lo_y), hi_y)
                for tx, ty, v, reach in th:
                    score = min(score, math.hypot(x - tx, y - ty) - v * n - reach)
                if q is me and any(math.hypot(x - v.x, y - v.y) < v.size + me.size for v in poppers):
                    score = -math.inf
        score += 1e-3 * (ux * pref[0] + uy * pref[1])
        if score > best_score:
            best, best_score = (ux, uy), score
    return best


def steer(s: Snapshot, plan: Plan, router: FoodRouter | None = None,
          p: Params = DEFAULT) -> tuple[float, float, bool]:
    """Return world aim point and whether to split this tick.

    With a router, food is approached along a planned pellet route; without one,
    every pellet pulls (the original potential-field behaviour).
    """
    me = s.main
    food, prey, threats, viruses = classify(s, p)
    w_food, w_prey, w_threat = {"farm": (1, p.farm_prey, p.farm_threat),
                                "hunt": (p.hunt_food, 1, p.hunt_threat),
                                "flee": (0.2, 0, p.flee_threat),
                                "hide": (0.2, 0, p.flee_threat * 0.75)}[plan.mode]
    fx = fy = 0.0

    def pull(c: Cell, weight: float) -> None:
        nonlocal fx, fy
        dx, dy = c.x - me.x, c.y - me.y
        d = max(math.hypot(dx, dy), 1)
        fx += weight * dx / d
        fy += weight * dy / d

    if router is None:
        for c in food:
            pull(c, w_food * 40 / (me.dist(c) + 40))
    else:
        poppable = [v for v in viruses if me.mass > v.mass * EAT_RATIO and p.virus_detour > 0.5]
        # Unsafe pellets: in a threat's reach, or only reachable by touching a virus we'd pop on.
        unsafe = lambda c: (any(t.dist(c) < danger_radius(t, me, p, s.border) for t in threats)
                            or any(v.dist(c) < v.size + me.size for v in poppable))
        wx, wy = router.next_waypoint(s, food, unsafe, p.switch_margin)
        pull(Cell(-1, wx, wy, 0, "waypoint", ""), w_food * p.food_pull)
    def ahead(c: Cell) -> Cell:
        """Where `c` will be: lead its motion, more for far targets (0 = no lead)."""
        if not p.lead:
            return c
        k = p.lead * min(1.0, me.dist(c) / 600)
        return Cell(c.id, c.x + c.vx * k, c.y + c.vy * k, c.size, c.kind, c.name)

    for c in prey:
        bonus = p.target_bonus if c.id == plan.target_id else 1
        pull(ahead(c), w_prey * bonus * c.mass / (me.dist(c) + 50))
    for c in threats:
        d = me.dist(c) - c.size - me.size
        r = danger_radius(c, me, p, s.border)
        if d < r:
            pull(c, -w_threat * p.threat_force * (r - d) / r)
    if p.piece_threats > 0.5 and len(s.mine) > 1:   # our other pieces' threats, by their mass share
        total = sum(q.mass for q in s.mine) or 1.0
        mine = {q.id for q in s.mine}
        for q in s.mine:
            if q is me:
                continue
            for c in s.others:
                if c.kind != "player" or c.id in mine or c.mass <= q.mass * EAT_RATIO:
                    continue
                d = q.dist(c) - c.size - q.size
                r = danger_radius(c, q, p, s.border)
                if d < r:
                    dx, dy = q.x - c.x, q.y - c.y
                    n = max(math.hypot(dx, dy), 1)
                    w = w_threat * p.threat_force * (r - d) / r * q.mass / total
                    fx += w * dx / n
                    fy += w * dy / n
    poppers = []  # viruses we'd pop on: steered around below, not pushed against
    for v in viruses:
        if me.mass > v.mass * EAT_RATIO:
            if p.virus_detour > 0.5:
                poppers.append(v)
            else:  # old behaviour (kept for comparisons): push straight away
                d = me.dist(v) - v.size - me.size
                if d < p.virus_avoid:
                    pull(v, -20 * (p.virus_avoid - d) / p.virus_avoid)
        elif plan.mode == "hide":
            pull(v, 8)

    b = s.border  # push back from map edges
    margin = me.size + p.border_margin
    # ...but not from a wall our hunt target is closer to than we are: prey tucked into a
    # corner balanced the push against the prey pull and we shook there, never eating it.
    target = next((c for c in prey if c.id == plan.target_id), None) if plan.mode == "hunt" else None
    t_edges = ((target.x - b["left"], b["right"] - target.x, target.y - b["top"], b["bottom"] - target.y)
               if target else (math.inf,) * 4)
    for edge, t_edge, push in zip((me.x - b["left"], b["right"] - me.x, me.y - b["top"], b["bottom"] - me.y),
                                  t_edges, ((1, 0), (-1, 0), (0, 1), (0, -1))):
        if edge < margin and t_edge >= edge:
            fx += push[0] * 10 * (margin - edge) / margin
            fy += push[1] * 10 * (margin - edge) / margin

    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        fx, fy, norm = 1.0, 0.0, 1.0  # nothing visible: wander
    ux, uy = fx / norm, fy / norm
    planned = plan.mode == "flee" and p.flee_plan > 0.5 and (bool(threats) or p.flee_pieces > 0.5)
    if planned:
        ux, uy = flee_heading(s, me, threats, poppers, p, (ux, uy))
    # Viruses as obstacles: if the nearest one we'd pop on lies across our path, head
    # for the point beside it on the side we were already leaning to. Pushing straight
    # away from it (as before) balanced against the food pull and froze the cell.
    blocking = []
    for v in (() if planned else poppers):
        vx, vy = v.x - me.x, v.y - me.y
        t = vx * ux + vy * uy                        # distance ahead along our path
        clear = v.size + me.size + p.virus_avoid * 0.2
        h = vx * uy - vy * ux                        # signed sideways offset of the virus
        if 0 < t < clear + p.virus_avoid + 200 and abs(h) < clear:
            blocking.append((t, v, h, clear))
    if blocking:
        _, v, h, clear = min(blocking, key=lambda b: b[0])
        # h > 0: virus right of our path -> pass it on the left (and vice versa), the
        # smaller turn. (-uy, ux) is our left.
        side = 1.0 if h > 0 else -1.0
        px, py = -uy * side, ux * side
        tx, ty = v.x + px * clear * 1.15, v.y + py * clear * 1.15
        dx, dy = tx - me.x, ty - me.y
        dn = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dn, dy / dn
    aim = (me.x + ux * AIM_DISTANCE, me.y + uy * AIM_DISTANCE)

    split = False
    if plan.split_ok and plan.mode == "hunt" and len(s.mine) <= round(p.split_max_cells):
        target = next((c for c in prey if c.id == plan.target_id), None)
        target = ahead(target) if target else None
        side = half_of(me) if p.half_split_safety > 0.5 else me
        if (target and me.mass / 2 > target.mass * EAT_RATIO
                and me.dist(target) - me.size < p.split_range * p.split_reach
                and not any(me.dist(t) < danger_radius(t, side, p) + p.split_range
                            for t in split_threats(s, me, p))):
            aim = (target.x, target.y)
            split = True
    return aim[0], aim[1], split
