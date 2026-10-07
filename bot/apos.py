"""Native port of Apostolique's AposBot decision logic (MIT, github.com/Apostolique/Agar.io-bot).

Port of module 9 of vendor/apos/aposbot.user.js (`createAposBot(api)`): it scores 64 headings
(food / prey / viruses / walls / room to escape against threat zones with split reach), keeps the
heading it has unless another is clearly better, commits to a food or prey target inside a cone, and on
top of that flees, baits, hides in viruses, lines up virus feeds at big players, splits onto prey and
splits out of a pin. Constants, formulas and branch order follow the JS; comments are shortened, see
the original for the reasoning behind each number.

The Rust twin (rust/arena_client/src/apos.rs) is decision-identical: keep the two in step, and re-run
`arena.rust_tools parity` on a dump after touching either.

Adaptation to our world (the only deviations, all forced by what a `Snapshot` carries):
  * time: the JS reads wall-clock `getLastUpdate()`; here `now` advances `step_ms` per decision (80 ms
    for the lockstep `sync=2`, 40 ms ticks).
  * cell colour (grouping one player's pieces, telling ejected mass apart) is the protocol colour in
    `Cell.color`; names are not used, so a player cell is never a "pellet" (a pellet is `kind == food`).
  * `cell.ejected` (JS: unset by the headless shim, set in the page) is `kind == ejected`.
  * `getMemoryCells()` (cells that left view in the last 3 s) is rebuilt here from past snapshots; a
    remembered cell that should be inside the view box but is missing is dropped (eaten / popped).
  * `verticalDistance()` is the view-box diagonal the server would send, 1920x1080 / scale.
  * the screen-scale `Math.hypot` calls are plain `sqrt(dx*dx+dy*dy)`; `Math.round` is floor(x+.5).
  * splits / ejects are returned as flags of the decision; the JS fires them on the pointer of the
    previous tick (so it checks alignment against it), here they fire toward the returned aim point.
  * `getCells()` order is ascending cell id (JS integer-keyed object order); own cells are sorted by
    size descending, ties by id.
  * drawing calls are dropped.
"""

import math

# ----------------------------------------------------------------- the rules
EAT_RATIO = 1.25
PREY_RATIO = 1.4
THREAT_RATIO = 1.05
SPLIT_THREAT_RATIO = 2.6
SPLIT_KILL_RATIO = 1.6
SPLIT_REACH = 710
SPLIT_MIN_MASS = 35
MAX_CELLS = 16
VIRUS_MASS = 100
EJECT_COST = 18
EJECT_REACH = 660
VIRUS_FEED = 8
PELLET_MASS = 25
CASCADE_RANGE = 900
FORESEE_RANGE = 200
FEED_RANGE = 300
FEED_MIN = 26
FEED_SHARE = 0.7
FEEDER_RANGE = 700

# ------------------------------------------------------------- the weighting
FOOD_GAIN = 1000
FOOD_SOFT = 400
PREY_GAIN = 2.2
VIRUS_FOOD_GAIN = 0.5
SHELTER_GAIN = 0.8
THREAT_WEIGHT = 500
VIRUS_WEIGHT = 220
WALL_WEIGHT = 260
THREAT_MARGIN = 120
THREAT_BUFFER = 250
THREAT_LOOKAHEAD = 400
CLOSING_LEAD = 2000
SPLIT_FEAR = 0.45
SPLIT_BITE_FULL = 0.12
SPLIT_BITE_NONE = 0.03
SPLIT_FEAR_FLOOR = 0.2
SPLIT_SHUN = 0.4
ESCAPE_LEAN = 0.15
FLIGHT_SPEED = 0.5
FLIGHT_CROSS = 0.5
COVER_MARGIN = 60
COVER_DISCOUNT = 0.5
VIRUS_MARGIN = 60
VIRUS_BUFFER = 450
SHOT_WEIGHT = 2
SHOT_REACH = 900
WALL_LOOKAHEAD = 500
MAP_MARGIN = 20
WALL_TOUCH = 40
SLIDE_LEAN = 0.15
WALL_NEAR = 250
WALL_SHOVE = 90
WALL_PRESSURE = 2
ROAM_GAIN = 60
ROAM_RANGE = 4000
MENACE_RANGE = 1600
ESCAPE_GAIN = 90
ESCAPE_REACH = 3000
ESCAPE_BOX = 6000
FLEE_PRESSURE = 0.35
GATHER_PRESSURE = 0.1
MERGE_PULL = 0.35
CLEAR_MARGIN = 50
TURN_MARGIN = 6
TURN_MARGIN_SHARE = 0.08
TURN_REVERSAL = 0.15
COMMIT_CONE = 0.82
COMMIT_KEEP = 0.3
COMMIT_SWITCH = 1.35
COMMIT_LEAN = 3

# ------------------------------------------------------------------ tactics
SPLIT_MAX_CELLS = 2
SPLIT_COOLDOWN = 900
SPLIT_ALIGN = 0.97
SPLIT_TRUST = 0.9
SPLIT_CHASE = 0.5
SPLIT_WALK = 1500
PREY_FLEEING = 0.03
GRAZE_LEAD = 1000
GRAZE_DISCOUNT = 0.3
PREY_REACH = 1200
STRIKE_MARGIN = 200
TRAP_RANGE = 350
TRAP_SPAN = 4
TRAP_STEP = math.pi / 12
TRAP_BLOCKED = 6
SPLIT_WORTH = 0.06
SPLIT_CAUTION = 300
AGGRO_PRESSURE = 0.35
ESCAPE_CONTACT = 0.75
ESCAPE_CLEAR = 60
ESCAPE_SWALLOW = 0.5
ESCAPE_LOSING = 1.2
ESCAPE_AWAY = 0.3
ESCAPE_INTO = 0.5
CHASE_STRIDE = 25
POP_MIN_MASS = 280
POP_MIN_ENEMY = 150
POP_RANGE = 900
POP_WALK = 1600
POP_STANDOFF = 90
POP_ARC = 0.6
POP_STAGE_TIMEOUT = 8000
POP_STAGE_REST = 8000
HIDE_RANGE = 1600
LURE_WALK = 900
LURE_RANGE = 1400
LURE_BAIL = 90
LURE_CLEAR = 40
LURE_TIMEOUT = 5000
LURE_REST = 15000
LURE_CLOSING = 0.05
HUNT_RANGE = 700
HUNT_CLOSING = 0.45
HUNT_SUSTAIN = 600
HIDE_HOLD = 1500
POP_ALIGN = 0.978
POP_CONTACT = 0.2
POP_CONTACT_HOLD = 0.4
FEED_MEMORY = 30000
EJECT_INTERVAL = 120
POP_TIMEOUT = 6000
POP_SCAN = 400
BAIT_WINDOW = 4000
BAIT_RANGE = 1400
BAIT_MARGIN = 1.35
BAIT_DISCOUNT = 0.15
POUR_MERGE_FAR = 10000
POUR_REACH = 600
POUR_KEEP = SPLIT_MIN_MASS + EJECT_COST
POUR_PRESSURE = 0.2
POUR_TIMEOUT = 4000
POUR_REST = 6000
POUR_WORTH = 0.1

# ---------------------------------------------------------------- directions
DIRECTIONS = 64
DIR_STEP = (math.pi * 2) / DIRECTIONS
DIR_X = [math.cos(i * DIR_STEP) for i in range(DIRECTIONS)]
DIR_Y = [math.sin(i * DIR_STEP) for i in range(DIRECTIONS)]
FOOD_CONE = 11
FOOD_KERNEL = [math.cos(j * DIR_STEP) * math.cos(j * DIR_STEP) * math.cos(j * DIR_STEP) for j in range(FOOD_CONE + 1)]

# adaptation constants
MEMORY_MS = 3000
VIEW_BASE_X, VIEW_BASE_Y = 1920.0, 1080.0
MIN_SCALE = 0.15
INF = math.inf


def mass_of(c):
    return (c.size * c.size) / 100


def size_of(mass):
    return math.sqrt(mass * 100)


def walk_speed(size):
    return 2.2 * math.pow(size, -0.45)


def clamp01(v):
    return 0 if v < 0 else 1 if v > 1 else v


def split_appetite(meal, mass):
    bite = clamp01((meal / mass - SPLIT_BITE_NONE) / (SPLIT_BITE_FULL - SPLIT_BITE_NONE))
    return SPLIT_FEAR_FLOOR + (1 - SPLIT_FEAR_FLOOR) * bite


def span(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def merge_delay(mass):
    agario = 30000 + 23.3 * mass
    ogar = max(30000, size_of(mass) * 200)
    return max(agario, ogar)


def distance_to_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / length if length > 0 else 0
    if t < 0:
        t = 0
    elif t > 1:
        t = 1
    return span(px, py, ax + dx * t, ay + dy * t)


def js_round(v):
    return math.floor(v + 0.5)


class C:
    """A cell as the bot sees it (the JS api cell)."""
    __slots__ = ("id", "x", "y", "size", "color", "virus", "ejected", "kind")

    def __init__(self, id, x, y, size, color, kind):
        self.id, self.x, self.y, self.size, self.color, self.kind = id, x, y, size, color, kind
        self.virus = kind == "virus"
        self.ejected = kind == "ejected"


class E:
    """An enemy entry (survey's object): a player cell, or a merged pair of one colour (cell None)."""
    __slots__ = ("x", "y", "size", "mass", "id", "cell", "remembered", "potential", "grown", "absorbed",
                 "eats", "vx", "vy", "trapped", "catch")

    def __init__(self, x, y, size, mass, id, cell, remembered=False):
        self.x, self.y, self.size, self.mass, self.id, self.cell, self.remembered = x, y, size, mass, id, cell, remembered
        self.potential = 0
        self.grown = 0
        self.absorbed = None
        self.eats = 0
        self.vx = 0.0
        self.vy = 0.0
        self.trapped = False
        self.catch = None


class _Ahead:
    __slots__ = ("x", "y", "size", "vx", "vy")

    def __init__(self, x, y, size, vx, vy):
        self.x, self.y, self.size, self.vx, self.vy = x, y, size, vx, vy


class _Near:
    __slots__ = ("cell", "distance", "ux", "uy")

    def __init__(self, cell, distance, ux, uy):
        self.cell, self.distance, self.ux, self.uy = cell, distance, ux, uy


class _Motion:
    __slots__ = ("x", "y", "at", "vx", "vy", "moved")

    def __init__(self, x, y, at):
        self.x, self.y, self.at, self.vx, self.vy, self.moved = x, y, at, 0.0, 0.0, False


class _Pick:
    __slots__ = ("id", "x", "y", "size", "kind", "distance", "ux", "uy")


class _View:
    pass


class AposPolicy:
    """`act(snapshot) -> (x, y, split, eject)`, optional `reset()`; spec {"kind": "apos", ...}."""

    def __init__(self, aggressive: bool = True, step_ms: float = 80.0, memory_ms: float = MEMORY_MS):
        self.aggressive = bool(aggressive)
        self.step_ms = float(step_ms)
        self.memory_ms = float(memory_ms)
        self.counters = {"steps": 0, "splits": 0, "ejects": 0, "shots_started": 0, "shots_fired": 0, "virus_hides": 0,
                         "baits": 0, "lures": 0, "pours": 0, "escapes": 0, "double_splits": 0}
        self.now = 0.0
        self._memory: dict[int, tuple[C, float]] = {}
        self._init_state()
        self.point_x = 0.0
        self.point_y = 0.0
        self.heading = None
        self.slide_x = 1
        self.slide_y = 1
        self.pursuit: dict = {}
        self.stage_rest: dict = {}
        self.fed_viruses: dict = {}
        self.pour_rest = 0
        self.lure_rest = 0
        self.pop_scan = 0
        self.broke_at = 0
        self.motion: dict = {}
        self.split_at = 0
        self.risk = [0.0] * DIRECTIONS
        self.gain = [0.0] * DIRECTIONS
        self.prey_gain = [0.0] * DIRECTIONS
        self.value = [0.0] * DIRECTIONS
        self.repositioning = False
        self.menace = 0.0
        self.cell_count = 0
        self.contact = 0.0
        self.bait_id = None
        self.merge_left = 0.0
        self._split = False
        self._eject = False
        self._vd = 0.0

    def _init_state(self):
        self.state = "roam"
        self.detour = None
        self.hunted_at = 0
        self.being_chased = False
        self.target_id = None
        self.feeding = None
        self.staging = None
        self.pouring = None
        self.luring = None

    # ------------------------------------------------------------ interface

    def reset(self) -> None:
        """What mainLoop does with no cells of ours (death); everything else, incl. motion, persists."""
        self.now += self.step_ms
        self.state = "roam"
        self.cell_count = 0
        self.feeding = None
        self.staging = None
        self.luring = None
        self.pouring = None
        self.target_id = None
        self.detour = None
        self.hunted_at = 0
        self.pursuit.clear()

    def act(self, s):
        self._split = False
        self._eject = False
        self.now += self.step_ms
        mine = [C(c.id, c.x, c.y, c.size, c.color, "mine") for c in s.mine]
        mine.sort(key=lambda c: (-c.size, c.id))
        self._border = s.border
        total_size = 0.0
        for c in mine:
            total_size += c.size
        scale = max(math.pow(min(64 / total_size, 1), 0.4), MIN_SCALE)
        self._vd = math.sqrt(VIEW_BASE_X * VIEW_BASE_X + VIEW_BASE_Y * VIEW_BASE_Y) / scale
        self._scale = scale
        target = self.main_loop(mine, s)
        if target:
            self.point_x, self.point_y = target
        c = self.counters
        c["steps"] += 1
        c["splits"] += self._split
        c["ejects"] += self._eject
        k = "mode_" + self.state
        c[k] = c.get(k, 0) + 1
        x, y = (target if target else (self.point_x, self.point_y))
        return (x, y, self._split, self._eject)

    # --------------------------------------------------------- api helpers

    def _known(self):
        b = self._border
        return (math.isfinite(b["left"]) and math.isfinite(b["right"])
                and math.isfinite(b["top"]) and math.isfinite(b["bottom"]))

    def _min_x(self):
        return self._border["left"]

    def _max_x(self):
        return self._border["right"]

    def _min_y(self):
        return self._border["top"]

    def _max_y(self):
        return self._border["bottom"]

    def split(self):
        self._split = True

    def shoot(self):
        self._eject = True

    # ----------------------------------------------------------- the world

    def _cells(self, mine, s):
        """Visible cells plus remembered players / viruses, ascending id: [(C, remembered)]."""
        owned = {c.id for c in mine}
        visible = {}
        for o in s.others:
            if o.id in owned:
                continue
            visible[o.id] = C(o.id, o.x, o.y, o.size, o.color, o.kind)
        cx = sum(c.x for c in mine) / len(mine)
        cy = sum(c.y for c in mine) / len(mine)
        hw = VIEW_BASE_X / self._scale / 2 * 0.9
        hh = VIEW_BASE_Y / self._scale / 2 * 0.9
        mem = self._memory
        for cid, c in visible.items():
            if c.kind in ("player", "virus"):
                mem[cid] = (c, self.now)
        for cid in list(mem):
            if cid in visible:
                continue
            c, seen = mem[cid]
            if cid in owned or self.now - seen > self.memory_ms or (abs(c.x - cx) < hw and abs(c.y - cy) < hh):
                del mem[cid]
        out = [(c, False) for c in visible.values()]
        out += [(c, True) for cid, (c, _) in mem.items() if cid not in visible]
        out.sort(key=lambda t: t[0].id)
        return out

    def survey(self, mine, s):
        total = 0
        cx = 0
        cy = 0
        big = mine[0]
        small = mine[0]
        for cell in mine:
            total += mass_of(cell)
            cx += cell.x
            cy += cell.y
            if cell.size > big.size:
                big = cell
            if cell.size < small.size:
                small = cell
        cx /= len(mine)
        cy /= len(mine)
        spread = 0
        for cell in mine:
            spread = max(spread, span(cell.x, cell.y, cx, cy) + cell.size)
        big_mass = mass_of(big)
        small_mass = mass_of(small)
        food = []
        viruses = []
        enemies = []
        for cell, remembered in self._cells(mine, s):
            if cell.virus:
                viruses.append(cell)
                continue
            mass = mass_of(cell)
            eatable = mass * PREY_RATIO < big_mass
            if cell.ejected:
                if eatable and not remembered:
                    food.append(cell)
                continue
            if eatable and cell.kind == "food":
                if not remembered:
                    food.append(cell)
                continue
            enemies.append(E(cell.x, cell.y, cell.size, mass, cell.id, cell, remembered))
        self.foresee(enemies, food)
        ladder = sorted(({"id": c.id, "mass": mass_of(c), "x": c.x, "y": c.y, "size": c.size} for c in mine),
                        key=lambda e: e["mass"])
        for entry in enemies:
            self.cascade(entry, ladder)
        prey = []
        threats = []
        for entry in enemies:
            if not entry.remembered and entry.grown * PREY_RATIO < big_mass:
                prey.append(entry)
            if entry.grown > small_mass * THREAT_RATIO:
                threats.append(entry)
        self.merge_threats(threats, enemies, small_mass, ladder)
        zones = []
        for threat in threats:
            grown = self.grown_for(threat, small)
            splits = grown > small_mass * SPLIT_THREAT_RATIO
            mouth = threat.size + big.size + THREAT_MARGIN
            jump = mouth + SPLIT_REACH
            zones.append((threat.x, threat.y, mouth * mouth, jump * jump if splits else 0,
                          split_appetite(max(small_mass, threat.eats), threat.mass) if splits else 0))
        v = _View()
        v.mine = mine
        v.centre = (cx, cy)
        v.spread = spread
        v.big = big
        v.small = small
        v.big_mass = big_mass
        v.small_mass = small_mass
        v.total = total
        v.food = food
        v.prey = prey
        v.threats = threats
        v.viruses = viruses
        v.enemies = enemies
        v.zones = zones
        return v

    def cascade(self, entry, ladder):
        grown = entry.mass + (entry.potential or 0)
        absorbed = None
        eats = 0
        for piece in ladder:
            if grown < piece["mass"] * EAT_RATIO:
                break
            gap = span(entry.x, entry.y, piece["x"], piece["y"]) - entry.size - piece["size"]
            if gap > CASCADE_RANGE:
                continue
            grown += piece["mass"]
            eats += piece["mass"]
            if absorbed is None:
                absorbed = []
            absorbed.append(piece["id"])
        entry.grown = grown
        entry.absorbed = absorbed
        entry.eats = eats

    def foresee(self, enemies, food):
        ejected = [c for c in food if c.ejected]
        for entry in enemies:
            extra = 0
            for other in enemies:
                if other is entry:
                    continue
                if other.cell.color == entry.cell.color:
                    continue
                if entry.mass < other.mass * EAT_RATIO:
                    continue
                gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size
                reach = (FORESEE_RANGE + SPLIT_REACH) if entry.mass >= other.mass * SPLIT_THREAT_RATIO else FORESEE_RANGE
                if gap <= reach:
                    extra += other.mass
            fed = 0
            for cell in ejected:
                if cell.color == entry.cell.color:
                    continue
                gap = span(entry.x, entry.y, cell.x, cell.y) - entry.size
                if gap <= FEED_RANGE:
                    fed += mass_of(cell)
            if fed >= FEED_MIN:
                extra += fed
                feeder = None
                closest = FEEDER_RANGE
                for other in enemies:
                    if other is entry:
                        continue
                    if other.cell.color == entry.cell.color:
                        continue
                    gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size
                    if gap < closest:
                        closest = gap
                        feeder = other
                if feeder:
                    extra += feeder.mass * FEED_SHARE
            entry.potential = extra

    def grown_for(self, entry, cell):
        if not entry.absorbed:
            return entry.grown
        return entry.grown if cell.id not in entry.absorbed else entry.grown - mass_of(cell)

    def merge_threats(self, threats, enemies, small_mass, ladder):
        groups: dict = {}
        for entry in enemies:
            if entry.cell is None:
                continue
            groups.setdefault(entry.cell.color, []).append(entry.cell)
        for group in groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    reach = a.size + b.size
                    if abs(a.x - b.x) > reach or abs(a.y - b.y) > reach:
                        continue
                    if span(a.x, a.y, b.x, b.y) > reach * 0.9:
                        continue
                    mass = mass_of(a) + mass_of(b)
                    entry = E((a.x + b.x) / 2, (a.y + b.y) / 2, size_of(mass), mass, None, None)
                    self.cascade(entry, ladder)
                    if entry.grown <= small_mass * THREAT_RATIO:
                        continue
                    threats.append(entry)

    def track_motion(self, view):
        now = self.now
        seen = set()
        motion = self.motion

        def remember(cid, x, y, entry):
            seen.add(cid)
            last = motion.get(cid)
            if last is None:
                motion[cid] = _Motion(x, y, now)
                return
            if now > last.at:
                elapsed = now - last.at
                vx = (x - last.x) / elapsed
                vy = (y - last.y) / elapsed
                last.vx = (last.vx + vx) / 2 if last.moved else vx
                last.vy = (last.vy + vy) / 2 if last.moved else vy
                last.moved = True
                last.x = x
                last.y = y
                last.at = now
            if entry is not None:
                entry.vx = last.vx
                entry.vy = last.vy

        for entry in view.prey:
            if entry.cell is not None:
                remember(entry.cell.id, entry.x, entry.y, entry)
        for entry in view.threats:
            if entry.cell is not None:
                remember(entry.cell.id, entry.x, entry.y, entry)
        for cell in view.mine:
            remember(cell.id, cell.x, cell.y, None)
        for cid in list(motion):
            if cid not in seen:
                del motion[cid]

    def speed_of(self, cid):
        last = self.motion.get(cid)
        if last is None:
            return 0
        return math.sqrt(last.vx * last.vx + last.vy * last.vy)

    # ------------------------------------------------------------- scoring

    def score(self, view):
        self.risk = [0.0] * DIRECTIONS
        self.gain = [0.0] * DIRECTIONS
        self.prey_gain = [0.0] * DIRECTIONS
        risk, gain, prey_gain = self.risk, self.gain, self.prey_gain
        pressure = self.score_threats(view, risk)
        self.score_food(view, gain)
        self.score_prey(view, prey_gain)
        self.score_viruses(view, risk, gain, pressure)
        self.score_walls(view, risk, pressure)
        self.score_open(view, gain)
        self.score_escape(view, gain)
        return pressure

    def score_threats(self, view, risk):
        pressure = 0
        self.contact = 0
        horizon = self.horizon()
        for threat in view.threats:
            baited = threat.id is not None and threat.id == self.bait_id
            discount = BAIT_DISCOUNT if baited else 1
            ahead = self.ahead(threat)
            for cell in view.mine:
                mass = mass_of(cell)
                grown = self.grown_for(threat, cell)
                if grown < mass * THREAT_RATIO:
                    continue
                weight = discount * COVER_DISCOUNT if self.covered(view, cell, threat) else discount
                reach = threat.size + cell.size + THREAT_MARGIN
                splits = grown > mass * SPLIT_THREAT_RATIO and grown >= SPLIT_MIN_MASS
                fear = SPLIT_FEAR * split_appetite(max(mass, threat.eats), threat.mass) if splits else 0
                gap = min(span(threat.x, threat.y, cell.x, cell.y), span(ahead.x, ahead.y, cell.x, cell.y)) - reach
                near = clamp01(1 - gap / THREAT_BUFFER)
                jump = clamp01(1 - (gap - SPLIT_REACH) / THREAT_BUFFER) if splits else 0
                if near > 0 or jump > 0:
                    pressure = max(pressure, weight * max(near, jump * fear))
                    self.contact = max(self.contact, weight * near)
                stake = mass / view.big_mass
                nearer = max(0, self.closing_on(threat, cell)) * CLOSING_LEAD
                self.sweep(risk, cell, ahead, reach, THREAT_BUFFER, weight * stake * THREAT_WEIGHT, horizon, nearer)
                if splits:
                    self.sweep(risk, cell, ahead, reach + SPLIT_REACH, THREAT_BUFFER,
                               weight * stake * THREAT_WEIGHT * fear, horizon, nearer)
        return pressure

    def ahead(self, threat):
        if not threat.vx and not threat.vy:
            return threat
        return _Ahead(threat.x + threat.vx * THREAT_LOOKAHEAD, threat.y + threat.vy * THREAT_LOOKAHEAD,
                      threat.size, threat.vx, threat.vy)

    def sweep(self, risk, cell, thing, radius, buffer, weight, horizon, nearer=0):
        rx = thing.x - cell.x
        ry = thing.y - cell.y
        distance = math.sqrt(rx * rx + ry * ry)
        if distance > horizon + radius + buffer:
            return
        if self.in_flight(thing, rx, ry):
            self.sweep_flight(risk, thing.x, thing.y, thing.vx, thing.vy, rx, ry, radius, buffer, weight)
            return
        band = radius + buffer
        for i in range(DIRECTIONS):
            along = rx * DIR_X[i] + ry * DIR_Y[i]
            if along <= 0:
                clear = distance - radius
                if clear >= buffer:
                    continue
                close = 1 - clear / buffer if clear > 0 else 1
                leaving = -along / distance
                risk[i] -= weight * close * close * leaving * ESCAPE_LEAN
                continue
            due = along - nearer if along > nearer else 0
            if due >= horizon:
                continue
            cross = rx * DIR_Y[i] - ry * DIR_X[i]
            clearance = -cross if cross < 0 else cross
            gap = clearance - radius
            if gap >= buffer:
                continue
            near = 1 - gap / buffer if gap > 0 else 1
            closing = clamp01((distance - clearance) / band)
            soon = 1 - due / horizon
            risk[i] += weight * near * near * closing * soon

    def in_flight(self, thing, rx, ry):
        vx = getattr(thing, "vx", 0) or 0
        vy = getattr(thing, "vy", 0) or 0
        if vx * vx + vy * vy < FLIGHT_SPEED * FLIGHT_SPEED:
            return False
        return rx * vx + ry * vy < 0

    def sweep_flight(self, risk, tx, ty, tvx, tvy, rx, ry, radius, buffer, weight):
        speed = math.sqrt(tvx * tvx + tvy * tvy)
        vx = tvx / speed
        vy = tvy / speed
        clear = abs(rx * vy - ry * vx) - radius
        if clear >= buffer:
            return
        close = 1 - clear / buffer if clear > 0 else 1
        toward = 1 if ry * vx - rx * vy > 0 else -1
        for i in range(DIRECTIONS):
            along = DIR_X[i] * vx + DIR_Y[i] * vy
            across = (DIR_Y[i] * vx - DIR_X[i] * vy) * toward
            cost = along * along
            if across > 0:
                cost += FLIGHT_CROSS * across * across
            risk[i] += weight * close * cost

    def covered(self, view, cell, threat):
        if len(view.mine) < 2:
            return False
        if view.big_mass < threat.grown * PREY_RATIO:
            return False
        dx = threat.x - cell.x
        dy = threat.y - cell.y
        away = math.sqrt(dx * dx + dy * dy) or 1
        bite = threat.size + cell.size
        mx = cell.x + (dx / away) * bite
        my = cell.y + (dy / away) * bite
        for own in view.mine:
            if own.id == cell.id:
                continue
            if mass_of(own) < threat.grown * PREY_RATIO:
                continue
            if span(own.x, own.y, mx, my) <= own.size + COVER_MARGIN:
                return True
        return False

    def horizon(self):
        return max(600, self._vd * 0.5)

    def score_food(self, view, gain):
        grazers = self.grazers(view)
        for cell in view.food:
            near = self.nearest_cell(view.mine, cell.x, cell.y)
            if near.distance < 1:
                continue
            value = (FOOD_GAIN * mass_of(cell)) / (near.distance + FOOD_SOFT)
            if grazers and self.grazed(grazers, cell):
                value *= GRAZE_DISCOUNT
            self.spread(gain, near.ux, near.uy, value)

    def grazers(self, view):
        out = []
        for entry in view.prey:
            speed = math.sqrt(entry.vx * entry.vx + entry.vy * entry.vy)
            if speed <= 0:
                continue
            out.append((entry.x, entry.y, entry.vx / speed, entry.vy / speed, entry.size + speed * GRAZE_LEAD))
        return out

    def grazed(self, grazers, cell):
        for gx, gy, ux, uy, reach in grazers:
            dx = cell.x - gx
            dy = cell.y - gy
            if dx * ux + dy * uy <= 0:
                continue
            if dx * dx + dy * dy < reach * reach:
                return True
        return False

    def score_prey(self, view, prey_gain):
        ours = self.our_speed(view)
        for entry in view.prey:
            near = self.nearest_cell(view.mine, entry.x, entry.y, entry.mass)
            if near.cell is None or near.distance < 1:
                continue
            entry.trapped = self.cornered(entry) or (near.distance < PREY_REACH and self.trapped(view, entry, near))
            entry.catch = self.catchable(view, entry, near, ours)
            value = (FOOD_GAIN * entry.mass * entry.catch) / (near.distance + FOOD_SOFT)
            if near.distance < near.cell.size + SPLIT_REACH and mass_of(near.cell) / 2 >= entry.grown * SPLIT_KILL_RATIO:
                value *= 2
            if entry.trapped:
                value *= 1.5
            if near.distance > PREY_REACH:
                value *= PREY_REACH / near.distance
            self.spread(prey_gain, near.ux, near.uy, value)

    def trapped(self, view, entry, near):
        pops = entry.mass > VIRUS_MASS * EAT_RATIO
        heading = math.atan2(near.uy, near.ux)
        blocked = 0
        for k in range(-TRAP_SPAN, TRAP_SPAN + 1):
            angle = heading + k * TRAP_STEP
            dx = math.cos(angle)
            dy = math.sin(angle)
            if self.room_along(entry, dx, dy) < TRAP_RANGE:
                blocked += 1
                continue
            if not pops:
                continue
            end_x = entry.x + dx * TRAP_RANGE
            end_y = entry.y + dy * TRAP_RANGE
            for virus in view.viruses:
                gap = distance_to_segment(virus.x, virus.y, entry.x, entry.y, end_x, end_y)
                if gap < virus.size:
                    blocked += 1
                    break
        return blocked >= TRAP_BLOCKED

    def room_along(self, cell, dx, dy):
        if not self._known():
            return INF
        min_x, max_x, min_y, max_y = self._min_x(), self._max_x(), self._min_y(), self._max_y()
        exit_ = INF
        if dx > 0:
            exit_ = (max_x - cell.size - cell.x) / dx
        elif dx < 0:
            exit_ = (min_x + cell.size - cell.x) / dx
        if dy > 0:
            exit_ = min(exit_, (max_y - cell.size - cell.y) / dy)
        elif dy < 0:
            exit_ = min(exit_, (min_y + cell.size - cell.y) / dy)
        return 0 if exit_ < 0 else exit_

    def catchable(self, view, entry, near, ours):
        away = entry.vx * near.ux + entry.vy * near.uy
        if away <= 0:
            return 1
        if entry.trapped:
            return 1
        if view.big_mass / 2 >= entry.grown * SPLIT_KILL_RATIO:
            half = size_of(view.big_mass / 2)
            reach = half + SPLIT_REACH * SPLIT_TRUST
            if span(entry.x, entry.y, view.big.x, view.big.y) < reach:
                return 1
        chaser = self.speed_of(near.cell.id) or ours
        if chaser <= 0:
            return 1
        return max(PREY_FLEEING, 1 - away / chaser)

    def score_viruses(self, view, risk, gain, pressure):
        room = max(0, MAX_CELLS - len(view.mine))
        shelter = self.being_chased and self.shelters(view)
        horizon = self.horizon()
        breaks = view.big_mass > VIRUS_MASS * EAT_RATIO
        for virus in view.viruses:
            near = self.nearest_cell(view.mine, virus.x, virus.y)
            if near.distance < 1:
                continue
            if breaks:
                if room == 0:
                    value = (VIRUS_FOOD_GAIN * FOOD_GAIN * VIRUS_MASS) / (near.distance + FOOD_SOFT)
                    self.spread(gain, near.ux, near.uy, value)
                    continue
                weight = VIRUS_WEIGHT * (room / MAX_CELLS)
                shot = self.loaded(view, virus)
                for cell in view.mine:
                    mass = mass_of(cell)
                    if mass <= VIRUS_MASS * EAT_RATIO:
                        continue
                    radius = virus.size + cell.size + VIRUS_MARGIN
                    share = (weight * mass) / view.big_mass
                    self.sweep(risk, cell, virus, radius, VIRUS_BUFFER, share, horizon)
                    if not shot:
                        continue
                    rx = virus.x - cell.x
                    ry = virus.y - cell.y
                    if rx * shot[0] + ry * shot[1] >= 0:
                        continue
                    if rx * rx + ry * ry > SHOT_REACH * SHOT_REACH:
                        continue
                    self.sweep_flight(risk, virus.x, virus.y, shot[0], shot[1], rx, ry, radius, VIRUS_BUFFER,
                                      share * SHOT_WEIGHT)
            elif shelter:
                value = (SHELTER_GAIN * pressure * FOOD_GAIN * VIRUS_MASS) / (near.distance + FOOD_SOFT)
                self.spread(gain, near.ux, near.uy, value)

    def loaded(self, view, virus):
        fx = 0
        fy = 0
        feeder = None
        for cell in view.food:
            if not cell.ejected or cell.color == view.big.color:
                continue
            gap = span(cell.x, cell.y, virus.x, virus.y) - virus.size
            if gap > FEED_RANGE:
                continue
            fx += virus.x - cell.x
            fy += virus.y - cell.y
            for entry in view.enemies:
                if entry.cell.color != cell.color:
                    continue
                if span(entry.x, entry.y, virus.x, virus.y) > EJECT_REACH:
                    continue
                feeder = entry
        if not fx and not fy:
            return None
        if feeder:
            fx = virus.x - feeder.x
            fy = virus.y - feeder.y
        length = math.sqrt(fx * fx + fy * fy) or 1
        return (fx / length, fy / length)

    def shelters(self, view):
        for threat in view.threats:
            if threat.mass > VIRUS_MASS * EAT_RATIO:
                return True
        return False

    def score_walls(self, view, risk, pressure):
        if not self._known():
            return
        min_x, min_y, max_x, max_y = self._min_x(), self._min_y(), self._max_x(), self._max_y()
        widen = 1 + pressure * WALL_PRESSURE
        lookahead = WALL_LOOKAHEAD * widen
        for cell in view.mine:
            inset = cell.size
            left = min_x + inset - cell.x
            right = max_x - inset - cell.x
            top = min_y + inset - cell.y
            bottom = max_y - inset - cell.y
            for i in range(DIRECTIONS):
                dx = DIR_X[i]
                dy = DIR_Y[i]
                exit_ = INF
                if dx > 0:
                    exit_ = right / dx
                elif dx < 0:
                    exit_ = left / dx
                if dy > 0:
                    exit_ = min(exit_, bottom / dy)
                elif dy < 0:
                    exit_ = min(exit_, top / dy)
                if exit_ >= lookahead:
                    continue
                if exit_ < 0:
                    exit_ = 0
                crowding = 1 - exit_ / lookahead
                risk[i] += WALL_WEIGHT * crowding * crowding
            near = (WALL_NEAR + cell.size / 2) * widen
            self.push_wall(risk, cell.x - min_x, -1, 0, near)
            self.push_wall(risk, max_x - cell.x, 1, 0, near)
            self.push_wall(risk, cell.y - min_y, 0, -1, near)
            self.push_wall(risk, max_y - cell.y, 0, 1, near)

    def score_escape(self, view, gain):
        self.menace = 0
        if not self._known():
            return
        min_x, min_y, max_x, max_y = self._min_x(), self._min_y(), self._max_x(), self._max_y()
        menace = 0
        for threat in view.threats:
            for cell in view.mine:
                mass = mass_of(cell)
                if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                    continue
                if self.covered(view, cell, threat):
                    continue
                gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size
                near = clamp01(1 - gap / MENACE_RANGE)
                if near > menace:
                    menace = near
        self.menace = menace
        if menace <= 0:
            return
        ax, ay = view.centre
        here = ax - min_x
        if max_x - ax < here:
            here = max_x - ax
        if ay - min_y < here:
            here = ay - min_y
        if max_y - ay < here:
            here = max_y - ay
        boxed = clamp01(1 - here / ESCAPE_BOX)
        if boxed <= 0:
            return
        weight = ESCAPE_GAIN * menace * boxed
        for i in range(DIRECTIONS):
            x = ax + DIR_X[i] * ESCAPE_REACH
            y = ay + DIR_Y[i] * ESCAPE_REACH
            room = x - min_x
            if max_x - x < room:
                room = max_x - x
            if y - min_y < room:
                room = y - min_y
            if max_y - y < room:
                room = max_y - y
            if room < 0:
                room = 0
            elif room > ESCAPE_REACH:
                room = ESCAPE_REACH
            gain[i] += (weight * room) / ESCAPE_REACH

    def score_open(self, view, gain):
        self.repositioning = False
        if view.food or view.prey:
            return
        if not self._known():
            return
        min_x, min_y, max_x, max_y = self._min_x(), self._min_y(), self._max_x(), self._max_y()
        self.repositioning = True
        cell = view.big
        for i in range(DIRECTIONS):
            x = cell.x + DIR_X[i] * ROAM_RANGE
            y = cell.y + DIR_Y[i] * ROAM_RANGE
            room = x - min_x
            if max_x - x < room:
                room = max_x - x
            if y - min_y < room:
                room = y - min_y
            if max_y - y < room:
                room = max_y - y
            if room < 0:
                room = 0
            elif room > ROAM_RANGE:
                room = ROAM_RANGE
            gain[i] += (ROAM_GAIN * room) / ROAM_RANGE

    def push_wall(self, risk, gap, ux, uy, near):
        if gap >= near:
            return
        crowding = 1 - gap / near if gap > 0 else 1
        weight = WALL_SHOVE * crowding * crowding
        for i in range(DIRECTIONS):
            dot = DIR_X[i] * ux + DIR_Y[i] * uy
            if dot > 0:
                risk[i] += weight * dot * dot

    def cornered(self, entry):
        if not (math.isfinite(self._min_x()) and math.isfinite(self._max_x())):
            return False
        margin = entry.size + 300
        return (entry.x - self._min_x() < margin or self._max_x() - entry.x < margin
                or entry.y - self._min_y() < margin or self._max_y() - entry.y < margin)

    def nearest_cell(self, mine, x, y, prey_mass=None):
        cell = None
        best = INF
        for own in mine:
            if prey_mass is not None and mass_of(own) < prey_mass * PREY_RATIO:
                continue
            dx = x - own.x
            dy = y - own.y
            distance = dx * dx + dy * dy
            if distance < best:
                best = distance
                cell = own
        if cell is None:
            return _Near(None, INF, 0, 0)
        distance = math.sqrt(best) or 1
        return _Near(cell, distance, (x - cell.x) / distance, (y - cell.y) / distance)

    def spread(self, field, ux, uy, value):
        b = js_round(math.atan2(uy, ux) / DIR_STEP)
        for j in range(-FOOD_CONE, FOOD_CONE + 1):
            index = (b + j + DIRECTIONS) % DIRECTIONS
            field[index] += value * FOOD_KERNEL[-j if j < 0 else j]

    # ------------------------------------------------------------ main loop

    def main_loop(self, mine, s):
        now = self.now
        self.target_pick = None
        if len(mine) > self.cell_count:
            self.broke_at = now
        self.cell_count = len(mine)
        view = self.survey(mine, s)
        self.track_motion(view)
        self.merge_left = self.merge_in(view, now)
        self.bait_id = self.bait_target(view, now)
        self.being_chased = self.chased(view, now)
        pressure = self.score(view)
        self.rank(view)
        for plan in (self.plan_pop, self.plan_escape, self.plan_lure):
            out = plan(view, now)
            if out:
                return out
        out = self.plan_hide(view)
        if out:
            return out
        out = self.plan_pour(view, now, pressure)
        if out:
            return out
        out = self.plan_split(view, now, pressure)
        if out:
            return out
        return self.steer(view, pressure)

    def plan_escape(self, view, now):
        if self.contact < ESCAPE_CONTACT:
            return None
        if now - self.split_at < SPLIT_COOLDOWN:
            return None
        if len(view.mine) >= MAX_CELLS:
            return None
        if view.big_mass < SPLIT_MIN_MASS * 2:
            return None
        cell = view.big
        mass = mass_of(cell)
        known = self._known()
        min_x, min_y, max_x, max_y = self._min_x(), self._min_y(), self._max_x(), self._max_y()
        pressing = None
        closest = INF
        for threat in view.threats:
            if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                continue
            gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size
            if gap < closest:
                closest = gap
                pressing = threat
        if pressing is None:
            return None
        away_x = cell.x - pressing.x
        away_y = cell.y - pressing.y
        away = math.sqrt(away_x * away_x + away_y * away_y) or 1
        ux = away_x / away
        uy = away_y / away
        ours = max(self.our_speed(view), walk_speed(cell.size))
        flying = self.closing_on(pressing, cell) > ours * ESCAPE_LOSING
        grown = self.grown_for(pressing, cell)
        splits = grown > mass * SPLIT_THREAT_RATIO and grown >= SPLIT_MIN_MASS
        if not splits and not flying:
            if self.run_open(view, cell):
                return None
        else:
            swallowed = flying and closest < cell.size * ESCAPE_SWALLOW
            boxed = True
            for i in range(DIRECTIONS):
                if DIR_X[i] * ux + DIR_Y[i] * uy <= ESCAPE_AWAY:
                    continue
                if self.room_along(cell, DIR_X[i], DIR_Y[i]) > SPLIT_REACH:
                    boxed = False
                    break
            if not boxed and not swallowed:
                return None
        half = size_of(view.big_mass / 2)
        best = None
        best_room = 0
        for i in range(DIRECTIONS):
            if DIR_X[i] * -ux + DIR_Y[i] * -uy > ESCAPE_INTO:
                continue
            x = cell.x + DIR_X[i] * SPLIT_REACH
            y = cell.y + DIR_Y[i] * SPLIT_REACH
            if known:
                if x < min_x + half or x > max_x - half or y < min_y + half or y > max_y - half:
                    continue
            worst = INF
            for threat in view.threats:
                gap = span(threat.x, threat.y, x, y) - threat.size - half - ESCAPE_CLEAR
                if gap < worst:
                    worst = gap
            if worst <= 0:
                continue
            if worst > best_room:
                best_room = worst
                best = i
        if best is None:
            return None
        at = self.point(view, DIR_X[best], DIR_Y[best], SPLIT_REACH)
        dx = at[0] - view.centre[0]
        dy = at[1] - view.centre[1]
        length = math.sqrt(dx * dx + dy * dy) or 1
        aim_x = self.point_x - view.centre[0]
        aim_y = self.point_y - view.centre[1]
        aim_length = math.sqrt(aim_x * aim_x + aim_y * aim_y) or 1
        aligned = (dx * aim_x + dy * aim_y) / (length * aim_length) > SPLIT_ALIGN
        if aligned:
            self.split()
            self.split_at = now
            self.state = "escape"
            self.counters["escapes"] += 1
        else:
            self.state = "flee"
        return self.aim(view, DIR_X[best], DIR_Y[best], SPLIT_REACH)

    def our_speed(self, view):
        fastest = 0
        for cell in view.mine:
            speed = self.speed_of(cell.id)
            if speed > fastest:
                fastest = speed
        return fastest

    def closing_on(self, threat, cell):
        last = self.motion.get(threat.id) if threat.id is not None else None
        if last is None:
            return 0
        dx = cell.x - threat.x
        dy = cell.y - threat.y
        away = math.sqrt(dx * dx + dy * dy) or 1
        return (last.vx * dx + last.vy * dy) / away

    def run_open(self, view, cell):
        mass = mass_of(cell)
        for i in range(DIRECTIONS):
            room = self.room_along(cell, DIR_X[i], DIR_Y[i])
            if room <= SPLIT_REACH:
                continue
            cut = False
            for threat in view.threats:
                if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                    continue
                if self.cuts_off(threat, cell, DIR_X[i], DIR_Y[i], room):
                    cut = True
                    break
            if not cut:
                return True
        return False

    def cuts_off(self, threat, cell, hx, hy, room):
        mouth = max(0, threat.size - cell.size / 3)
        pace = walk_speed(threat.size) / walk_speed(cell.size)
        run = min(room, ESCAPE_REACH)
        x = cell.x
        y = cell.y
        tx = threat.x
        ty = threat.y
        s = 0
        while s <= run:
            dx = x - tx
            dy = y - ty
            gap = math.sqrt(dx * dx + dy * dy)
            if gap <= mouth:
                return True
            if pace < 1 and (dx * hx + dy * hy) / gap > pace:
                return False
            x += hx * CHASE_STRIDE
            y += hy * CHASE_STRIDE
            tx += (dx / gap) * pace * CHASE_STRIDE
            ty += (dy / gap) * pace * CHASE_STRIDE
            s += CHASE_STRIDE
        return False

    def merge_in(self, view, now):
        if len(view.mine) < 2:
            return 0
        return max(0, self.broke_at + merge_delay(view.big_mass) - now)

    def bait_target(self, view, now):
        if len(view.mine) < 2 or not self.aggressive:
            return None
        if self.merge_in(view, now) > BAIT_WINDOW:
            return None
        best = None
        best_distance = INF
        for threat in view.threats:
            if threat.id is None:
                continue
            if threat.grown * BAIT_MARGIN > view.total:
                continue
            if threat.grown < view.small_mass * THREAT_RATIO:
                continue
            distance = span(threat.x, threat.y, view.centre[0], view.centre[1])
            if distance > BAIT_RANGE or distance >= best_distance:
                continue
            best = threat
            best_distance = distance
        return best.id if best else None

    def rank(self, view):
        value = self.value
        best = 0
        best_value = -INF
        low = INF
        for i in range(DIRECTIONS):
            value[i] = self.gain[i] + PREY_GAIN * self.prey_gain[i] - self.risk[i]
            if value[i] > best_value:
                best_value = value[i]
                best = i
            if value[i] < low:
                low = value[i]
        if self.heading is not None:
            steps = abs(best - self.heading)
            if steps > DIRECTIONS / 2:
                steps = DIRECTIONS - steps
            turning = (1 - math.cos(steps * DIR_STEP)) / 2
            margin = TURN_MARGIN + (TURN_MARGIN_SHARE + TURN_REVERSAL * turning) * (best_value - low)
            if value[self.heading] >= best_value - margin:
                best = self.heading
        self.heading = best
        return best

    def steer(self, view, pressure):
        best = self.heading
        chasing = PREY_GAIN * self.prey_gain[best] > self.gain[best]
        if self.bait_id is not None:
            self.state = "bait"
            self.counters["baits"] += 1
        elif pressure > FLEE_PRESSURE:
            self.state = "flee"
        elif chasing:
            self.state = "hunt"
        elif self.repositioning:
            self.state = "roam"
        elif self.gain[best] > 0:
            self.state = "feed"
        else:
            self.state = "roam"
        stride = self.stride(view, pressure)
        target = self.commit(view, DIR_X[best], DIR_Y[best])
        if target is None:
            clear = self.clear_heading(view, best, stride)
            return self.aim(view, DIR_X[clear[0]], DIR_Y[clear[0]], clear[1])
        reach = min(target.distance, stride)
        if not self.merging(view):
            reach = max(reach, self.floor(view, target.ux, target.uy))
        return self.aim(view, target.ux, target.uy, reach)

    def commit(self, view, ux, uy):
        best = None
        best_value = 0
        held = None
        held_value = 0
        cx, cy = view.centre

        def consider(id, x, y, size, worth, kind):
            nonlocal best, best_value, held, held_value
            dx = x - cx
            dy = y - cy
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 1:
                return
            along = (dx * ux + dy * uy) / distance
            keeping = id == self.target_id
            if along < (COMMIT_KEEP if keeping else COMMIT_CONE):
                return
            raw = worth() / (distance + FOOD_SOFT)
            if not keeping and raw <= best_value:
                return
            shun = self.shunned(view, x, y)
            if shun <= 0:
                return
            value = raw * shun * math.pow(along, COMMIT_LEAN)
            if not keeping and value <= best_value:
                return
            if self.crosses(view, x, y):
                return
            pick = _Pick()
            pick.id, pick.x, pick.y, pick.size, pick.kind, pick.distance = id, x, y, size, kind, distance
            pick.ux, pick.uy = dx / distance, dy / distance
            if keeping:
                held = pick
                held_value = value
            if value > best_value:
                best_value = value
                best = pick

        mouthful = self.mouthfuls(view)
        for cell in view.food:
            consider(cell.id, cell.x, cell.y, cell.size, lambda cell=cell: mouthful(cell), "ejected" if cell.ejected else "pellet")
        strike = view.big.size + SPLIT_REACH + STRIKE_MARGIN
        for entry in view.prey:
            caught = 1 if entry.catch is None else entry.catch
            if caught < 0.5:
                continue
            if span(entry.x, entry.y, view.big.x, view.big.y) > strike:
                continue
            consider(entry.id, entry.x, entry.y, entry.size, lambda entry=entry, caught=caught: entry.mass * PREY_GAIN * caught, "prey")
        pick = held if (held is not None and held_value * COMMIT_SWITCH >= best_value) else best
        self.target_id = pick.id if pick else None
        return pick

    def mouthfuls(self, view):
        radius = max(view.big.size, 1)
        buckets: dict = {}
        for cell in view.food:
            buckets.setdefault((math.floor(cell.x / radius), math.floor(cell.y / radius)), []).append(cell)
        reach = radius * radius

        def mouthful(cell):
            bx = math.floor(cell.x / radius)
            by = math.floor(cell.y / radius)
            mass = 0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    bucket = buckets.get((bx + i, by + j))
                    if not bucket:
                        continue
                    for other in bucket:
                        dx = other.x - cell.x
                        dy = other.y - cell.y
                        if dx * dx + dy * dy <= reach:
                            mass += mass_of(other)
            return mass
        return mouthful

    def shunned(self, view, x, y):
        shun = 1
        for zx, zy, r2, jump2, appetite in view.zones:
            dx = x - zx
            dy = y - zy
            reach = dx * dx + dy * dy
            if reach < r2:
                return 0
            if jump2 and reach < jump2:
                shun = min(shun, 1 - (1 - SPLIT_SHUN) * appetite)
        return shun

    def crosses(self, view, x, y):
        room = len(view.mine) < MAX_CELLS
        for cell in view.mine:
            mass = mass_of(cell)
            for threat in view.threats:
                if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                    continue
                reach = threat.size + cell.size + THREAT_MARGIN
                if self.through(cell, threat, reach, x, y):
                    return True
            if not room or mass <= VIRUS_MASS * EAT_RATIO:
                continue
            for virus in view.viruses:
                reach = virus.size + cell.size
                if self.through(cell, virus, reach, x, y):
                    return True
        return False

    def through(self, cell, thing, reach, x, y):
        gap = distance_to_segment(thing.x, thing.y, cell.x, cell.y, x, y)
        if gap >= reach:
            return False
        now = span(thing.x, thing.y, cell.x, cell.y)
        if now >= reach:
            return True
        return gap < now - 1

    def stride(self, view, pressure):
        floor = view.big.size + CLEAR_MARGIN
        if self.merging(view):
            return view.spread * MERGE_PULL
        gather = pressure > GATHER_PRESSURE or self.state == "bait"
        if len(view.mine) > 1 and gather:
            return max(view.spread + view.big.size, floor)
        return max(self._vd * 0.5, floor)

    def merging(self, view):
        return len(view.mine) > 1 and self.merge_left <= 0

    def floor(self, view, ux, uy):
        floor = 0
        for cell in view.mine:
            along = (cell.x - view.centre[0]) * ux + (cell.y - view.centre[1]) * uy
            edge = along + cell.size + CLEAR_MARGIN
            if edge > floor:
                floor = edge
        return floor

    def clear_heading(self, view, best, stride):
        reach = self.clear_reach(view, best, stride)
        if reach > 0:
            self.detour = None
            return (best, reach)
        if self.detour is not None:
            held = self.clear_reach(view, self.detour, stride)
            if held > 0:
                return (self.detour, held)
        value = self.value
        side = self.side_of(best)
        for step in range(1, DIRECTIONS // 2 + 1):
            up = (best + step) % DIRECTIONS
            down = (best - step + DIRECTIONS) % DIRECTIONS
            first = down if value[down] > value[up] else up
            if side < 0:
                first = down
            elif side > 0:
                first = up
            second = down if first == up else up
            for i in (first, second):
                got = self.clear_reach(view, i, stride)
                if got <= 0:
                    continue
                self.detour = i
                return (i, got)
        return (best, stride)

    def side_of(self, best):
        if self.detour is None:
            return 0
        step = self.detour - best
        if step > DIRECTIONS / 2:
            step -= DIRECTIONS
        elif step < -DIRECTIONS / 2:
            step += DIRECTIONS
        return 1 if step > 0 else -1 if step < 0 else 0

    def clear_reach(self, view, index, stride):
        far = self.point(view, DIR_X[index], DIR_Y[index], stride)
        if not self.crosses(view, far[0], far[1]):
            return stride
        floor = self.floor(view, DIR_X[index], DIR_Y[index])
        if floor >= stride:
            return 0
        near = self.point(view, DIR_X[index], DIR_Y[index], floor)
        if not self.crosses(view, near[0], near[1]):
            return floor
        return 0

    def aim(self, view, ux, uy, distance):
        return self.point(view, ux, uy, distance)

    def point(self, view, ux, uy, distance):
        x = view.centre[0] + ux * distance
        y = view.centre[1] + uy * distance
        if self._known():
            min_x, max_x, min_y, max_y = self._min_x(), self._max_x(), self._min_y(), self._max_y()
            low = MAP_MARGIN
            cell = view.big
            touch = cell.size + WALL_TOUCH
            block_x = (x < min_x + low and cell.x - min_x < touch) or (x > max_x - low and max_x - cell.x < touch)
            block_y = (y < min_y + low and cell.y - min_y < touch) or (y > max_y - low and max_y - cell.y < touch)
            if block_x and not block_y:
                if abs(uy) >= SLIDE_LEAN:
                    self.slide_y = -1 if uy < 0 else 1
                x = view.centre[0]
                y = view.centre[1] + self.slide_y * distance
            elif block_y and not block_x:
                if abs(ux) >= SLIDE_LEAN:
                    self.slide_x = -1 if ux < 0 else 1
                x = view.centre[0] + self.slide_x * distance
                y = view.centre[1]
            x = min(max(x, min_x + low), max_x - low)
            y = min(max(y, min_y + low), max_y - low)
        return (x, y)

    # ------------------------------------------------------------- tactics

    def plan_split(self, view, now, pressure):
        if not self.aggressive:
            return None
        if pressure > AGGRO_PRESSURE:
            return None
        if now - self.split_at < SPLIT_COOLDOWN:
            return None
        if len(view.mine) > SPLIT_MAX_CELLS:
            return None
        cell = view.big
        mass = view.big_mass
        if mass < SPLIT_MIN_MASS * 2:
            return None
        half = size_of(mass / 2)
        rng = half + SPLIT_REACH * SPLIT_TRUST
        ours = self.our_speed(view)
        best = None
        best_score = 0
        for entry in view.prey:
            if mass / 2 < entry.grown * SPLIT_KILL_RATIO:
                continue
            tx, ty = self.lead(entry, cell)
            distance = span(tx, ty, cell.x, cell.y)
            if distance > rng:
                continue
            if ours > 0 and self.speed_of(entry.id) < ours * SPLIT_CHASE and distance - cell.size < ours * SPLIT_WALK:
                continue
            if self.crosses(view, tx, ty):
                continue
            if not self.split_is_safe(view, cell, tx, ty):
                continue
            haul = self.haul(view, entry, tx, ty, half, mass)
            if haul < mass * SPLIT_WORTH:
                continue
            score = haul / (distance + 200)
            if score > best_score:
                best_score = score
                best = (tx, ty)
        if best is None:
            return None
        dx = best[0] - cell.x
        dy = best[1] - cell.y
        length = math.sqrt(dx * dx + dy * dy) or 1
        aim_x = self.point_x - cell.x
        aim_y = self.point_y - cell.y
        aim_length = math.sqrt(aim_x * aim_x + aim_y * aim_y) or 1
        aligned = (dx * aim_x + dy * aim_y) / (length * aim_length) > SPLIT_ALIGN
        if aligned:
            self.split()
            self.split_at = now
            self.state = "split"
        else:
            self.state = "hunt"
        return (cell.x + (dx / length) * rng, cell.y + (dy / length) * rng)

    def haul(self, view, entry, tx, ty, half, mass):
        total = entry.mass
        for other in view.prey:
            if other is entry:
                continue
            if mass / 2 < other.grown * SPLIT_KILL_RATIO:
                continue
            if span(other.x, other.y, tx, ty) < half:
                total += other.mass
        return total

    def lead(self, entry, cell):
        if not entry.vx and not entry.vy:
            return (entry.x, entry.y)
        distance = span(entry.x, entry.y, cell.x, cell.y)
        ahead = min(1000, (distance / SPLIT_REACH) * 1000)
        return (entry.x + entry.vx * ahead, entry.y + entry.vy * ahead)

    def split_is_safe(self, view, cell, tx, ty):
        half = mass_of(cell) / 2
        for enemy in view.enemies:
            if enemy.grown < half * THREAT_RATIO:
                continue
            splits = enemy.grown > half * SPLIT_THREAT_RATIO
            reach = enemy.size + cell.size + THREAT_MARGIN + SPLIT_CAUTION + (SPLIT_REACH if splits else 0)
            landing = span(enemy.x, enemy.y, tx, ty)
            behind = span(enemy.x, enemy.y, cell.x, cell.y)
            if landing < reach or behind < reach:
                return False
        if half > VIRUS_MASS * EAT_RATIO:
            dx = tx - cell.x
            dy = ty - cell.y
            length = math.sqrt(dx * dx + dy * dy) or 1
            reach = cell.size + SPLIT_REACH
            end_x = cell.x + (dx / length) * reach
            end_y = cell.y + (dy / length) * reach
            for virus in view.viruses:
                gap = distance_to_segment(virus.x, virus.y, cell.x, cell.y, end_x, end_y)
                if gap < virus.size + size_of(half) * 0.5:
                    return False
        return True

    # ---------------------------------------------------------- virus feeds

    def plan_pop(self, view, now):
        if not self.aggressive or len(view.mine) > 1:
            self.feeding = None
            self.staging = None
            return None
        if self.feeding:
            held = self.hold_pop(view, now)
            if held:
                return held
            self.feeding = None
        limit = POP_CONTACT_HOLD if self.staging else POP_CONTACT
        if self.contact > limit or view.total < POP_MIN_MASS + VIRUS_FEED * EJECT_COST or view.big_mass < SPLIT_MIN_MASS * 2:
            self.rest_stage(now)
            return None
        if self.staging:
            walk = self.hold_station(view, now)
            if walk:
                return walk
            self.rest_stage(now)
        if now - self.pop_scan < POP_SCAN:
            return None
        self.pop_scan = now
        setup = self.find_pop(view, now)
        if setup:
            virus, enemy = setup
            self.feeding = {"virus": virus.id, "enemy": enemy.id, "fed": self.fed_into(virus.id, now), "last": 0,
                            "since": now}
            self.counters["shots_started"] += 1
            return self.hold_pop(view, now)
        station = self.find_station(view, now)
        if not station:
            return None
        virus, enemy = station
        self.staging = {"virus": virus.id, "enemy": enemy.id, "since": now}
        walk = self.hold_station(view, now)
        if walk:
            return walk
        self.rest_stage(now)
        return None

    def rest_stage(self, now):
        if self.staging:
            self.stage_rest[self.staging["enemy"]] = now + POP_STAGE_REST
        self.staging = None
        self.feeding = None

    def pop_worthy(self, view, enemy, now):
        if enemy.id is None or enemy.mass < POP_MIN_ENEMY:
            return False
        if view.big_mass / 2 >= enemy.grown * SPLIT_KILL_RATIO:
            return False
        rest = self.stage_rest.get(enemy.id)
        if rest is None:
            return True
        if now < rest:
            return False
        del self.stage_rest[enemy.id]
        return True

    def hold_station(self, view, now):
        staging = self.staging
        if now - staging["since"] > POP_STAGE_TIMEOUT:
            return None
        virus = next((c for c in view.viruses if c.id == staging["virus"]), None)
        enemy = next((e for e in view.enemies if e.id == staging["enemy"]), None)
        if virus is None or enemy is None:
            return None
        cell = view.big
        if self.pop_line(cell, virus, enemy):
            self.feeding = {"virus": virus.id, "enemy": enemy.id, "fed": self.fed_into(virus.id, now), "last": 0,
                            "since": now}
            self.staging = None
            return self.hold_pop(view, now)
        if span(virus.x, virus.y, enemy.x, enemy.y) > POP_RANGE:
            return None
        stand = cell.size + virus.size + POP_STANDOFF
        if stand > EJECT_REACH:
            return None
        here = math.atan2(cell.y - virus.y, cell.x - virus.x)
        want = math.atan2(virus.y - enemy.y, virus.x - enemy.x)
        turn = want - here
        while turn > math.pi:
            turn -= math.pi * 2
        while turn < -math.pi:
            turn += math.pi * 2
        if turn > POP_ARC:
            turn = POP_ARC
        elif turn < -POP_ARC:
            turn = -POP_ARC
        for share in (1, 0.5, 0.25):
            angle = here + turn * share
            x = virus.x + math.cos(angle) * stand
            y = virus.y + math.sin(angle) * stand
            if self.shunned(view, x, y) <= 0:
                continue
            if self.crosses(view, x, y):
                continue
            self.state = "stage"
            return (x, y)
        return None

    def find_station(self, view, now):
        cell = view.big
        best = None
        best_score = 0
        for virus in view.viruses:
            walk = span(virus.x, virus.y, cell.x, cell.y)
            if walk > POP_WALK:
                continue
            if cell.size + virus.size + POP_STANDOFF > EJECT_REACH:
                continue
            if self.dud(virus, now):
                continue
            for enemy in view.enemies:
                if not self.pop_worthy(view, enemy, now):
                    continue
                to_enemy = span(virus.x, virus.y, enemy.x, enemy.y)
                if to_enemy > POP_RANGE or to_enemy < enemy.size:
                    continue
                score = enemy.mass / (walk + to_enemy + 400)
                if score > best_score:
                    best_score = score
                    best = (virus, enemy)
        return best

    def plan_lure(self, view, now):
        if not self.aggressive or len(view.mine) != 1 or now < self.lure_rest:
            self.luring = None
            return None
        cell = view.big

        def fits(virus):
            return cell.size >= virus.size and view.big_mass <= mass_of(virus) * EAT_RATIO

        virus = None
        if self.luring:
            if now - self.luring["since"] > LURE_TIMEOUT:
                self.lure_rest = now + LURE_REST
                self.luring = None
            else:
                virus = next((v for v in view.viruses if v.id == self.luring["virus"]), None)
                if virus is None:
                    self.luring = None
        if virus is None:
            virus = self.nearest_virus(view, LURE_WALK, fits)
        if virus is None or not fits(virus):
            self.luring = None
            return None
        bite = mass_of(virus) * EAT_RATIO
        mark = None
        closest = LURE_RANGE
        for threat in view.threats:
            if threat.mass <= bite:
                continue
            if self.closing_on(threat, cell) < LURE_CLOSING:
                continue
            gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size
            if gap >= closest:
                continue
            mark = threat
            closest = gap
        if mark is None:
            self.luring = None
            return None
        if not self.luring:
            self.luring = {"virus": virus.id, "since": now}
            self.counters["lures"] += 1
        self.state = "lure"
        if closest < LURE_BAIL:
            dx = virus.x - mark.x
            dy = virus.y - mark.y
            away = math.sqrt(dx * dx + dy * dy) or 1
            step = virus.size + cell.size + LURE_CLEAR
            return (virus.x + (dx / away) * step, virus.y + (dy / away) * step)
        return (virus.x, virus.y)

    def plan_hide(self, view):
        if view.big_mass > VIRUS_MASS * EAT_RATIO:
            return None
        if not self.being_chased:
            return None
        for threat in view.threats:
            for cell in view.mine:
                if self.grown_for(threat, cell) < mass_of(cell) * THREAT_RATIO:
                    continue
                if threat.mass <= VIRUS_MASS * EAT_RATIO:
                    return None
        best = self.nearest_virus(view, HIDE_RANGE)
        if best is None:
            return None
        self.state = "hide"
        self.counters["virus_hides"] += 1
        return (best.x, best.y)

    def plan_pour(self, view, now, pressure):
        if (not self.aggressive or len(view.mine) < 3 or self.merge_left < POUR_MERGE_FAR or now < self.pour_rest
                or pressure > POUR_PRESSURE or self.contact > 0):
            self.pouring = None
            return None
        big = view.big
        pour = 0
        for cell in view.mine:
            if cell.id == big.id:
                continue
            mass = mass_of(cell)
            if mass <= POUR_KEEP:
                continue
            gap = span(cell.x, cell.y, big.x, big.y) - cell.size - big.size
            if gap > POUR_REACH:
                continue
            pour += (mass - POUR_KEEP) * FEED_SHARE
        if pour < view.big_mass * POUR_WORTH:
            self.pouring = None
            return None
        purpose = self.pour_for(view, view.big_mass + pour)
        if purpose is None:
            self.pouring = None
            return None
        if not self.pouring:
            self.pouring = {"since": now, "last": 0}
            self.counters["pours"] += 1
        if now - self.pouring["since"] > POUR_TIMEOUT:
            self.pouring = None
            self.pour_rest = now + POUR_REST
            return None
        on_big = span(self.point_x, self.point_y, big.x, big.y) < big.size
        if on_big and now - self.pouring["last"] >= EJECT_INTERVAL:
            self.shoot()
            self.pouring["last"] = now
        self.state = "pour"
        return (big.x, big.y)

    def pour_for(self, view, poured):
        best = None
        closest = PREY_REACH
        for entry in view.enemies:
            distance = span(entry.x, entry.y, view.big.x, view.big.y)
            if distance >= closest:
                continue
            split_now = view.big_mass / 2 >= entry.grown * SPLIT_KILL_RATIO
            split_then = poured / 2 >= entry.grown * SPLIT_KILL_RATIO
            eat_now = view.big_mass >= entry.grown * PREY_RATIO
            eat_then = poured >= entry.grown * PREY_RATIO
            if (split_then and not split_now) or (eat_then and not eat_now):
                best = entry
                closest = distance
        return best

    def nearest_virus(self, view, rng, ok=None):
        cell = view.big
        best = None
        best_distance = rng
        for virus in view.viruses:
            distance = span(virus.x, virus.y, cell.x, cell.y)
            if distance >= best_distance:
                continue
            if ok and not ok(virus):
                continue
            if self.crosses(view, virus.x, virus.y):
                continue
            best = virus
            best_distance = distance
        return best

    def chased(self, view, now):
        ours = self.our_speed(view)
        pursuit = self.pursuit
        seen = set()
        chased = False
        for threat in view.threats:
            if threat.id is None:
                continue
            closing = False
            for cell in view.mine:
                mass = mass_of(cell)
                if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                    continue
                if self.covered(view, cell, threat):
                    continue
                gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size
                if gap > HUNT_RANGE:
                    continue
                if ours > 0 and self.closing_on(threat, cell) > ours * HUNT_CLOSING:
                    closing = True
                    break
            if not closing:
                continue
            seen.add(threat.id)
            since = pursuit.get(threat.id)
            if since is None:
                since = now
                pursuit[threat.id] = since
            if now - since >= HUNT_SUSTAIN:
                chased = True
        for tid in list(pursuit):
            if tid not in seen:
                del pursuit[tid]
        if chased:
            self.hunted_at = now
        return chased or (self.hunted_at > 0 and now - self.hunted_at < HIDE_HOLD and self.near_threat(view, HUNT_RANGE))

    def near_threat(self, view, rng):
        for threat in view.threats:
            for cell in view.mine:
                mass = mass_of(cell)
                if self.grown_for(threat, cell) < mass * THREAT_RATIO:
                    continue
                gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size
                if gap <= rng:
                    return True
        return False

    def hold_pop(self, view, now):
        feeding = self.feeding
        if now - feeding["since"] > POP_TIMEOUT:
            return None
        if feeding["fed"] >= VIRUS_FEED:
            return None
        if self.contact > POP_CONTACT_HOLD:
            return None
        if view.total < POP_MIN_MASS:
            return None
        virus = next((c for c in view.viruses if c.id == feeding["virus"]), None)
        enemy = next((e for e in view.enemies if e.id == feeding["enemy"]), None)
        if virus is None or enemy is None:
            return None
        cell = view.big
        line = self.pop_line(cell, virus, enemy)
        if not line:
            return None
        aim_x = self.point_x - cell.x
        aim_y = self.point_y - cell.y
        aim_length = math.sqrt(aim_x * aim_x + aim_y * aim_y) or 1
        aimed = (aim_x * line[0] + aim_y * line[1]) / aim_length >= POP_ALIGN
        if aimed and now - feeding["last"] >= EJECT_INTERVAL:
            self.shoot()
            feeding["fed"] += 1
            feeding["last"] = now
            self.fed_viruses[virus.id] = (feeding["fed"], now)
            if feeding["fed"] >= VIRUS_FEED:
                self.counters["shots_fired"] += 1
        self.state = "pop"
        hold = cell.size * 0.9
        return (cell.x + line[0] * hold, cell.y + line[1] * hold)

    def fed_into(self, virus_id, now):
        entry = self.fed_viruses.get(virus_id)
        if entry is None:
            return 0
        if now - entry[1] > FEED_MEMORY:
            del self.fed_viruses[virus_id]
            return 0
        return entry[0]

    def dud(self, virus, now):
        return self.fed_into(virus.id, now) >= VIRUS_FEED

    def find_pop(self, view, now):
        cell = view.big
        best = None
        best_align = POP_ALIGN
        for virus in view.viruses:
            if self.dud(virus, now):
                continue
            for enemy in view.enemies:
                if not self.pop_worthy(view, enemy, now):
                    continue
                line = self.pop_line(cell, virus, enemy)
                if not line or line[2] <= best_align:
                    continue
                best_align = line[2]
                best = (virus, enemy)
        return best

    def pop_line(self, cell, virus, enemy):
        dvx = virus.x - cell.x
        dvy = virus.y - cell.y
        to_virus = math.sqrt(dvx * dvx + dvy * dvy) or 1
        if to_virus > EJECT_REACH:
            return None
        if to_virus < cell.size + virus.size + 40:
            return None
        dex = enemy.x - virus.x
        dey = enemy.y - virus.y
        to_enemy = math.sqrt(dex * dex + dey * dey) or 1
        if to_enemy > POP_RANGE:
            return None
        gap = span(enemy.x, enemy.y, cell.x, cell.y)
        if gap < enemy.size + cell.size + 300:
            return None
        ux = dvx / to_virus
        uy = dvy / to_virus
        align = ux * (dex / to_enemy) + uy * (dey / to_enemy)
        if align < POP_ALIGN:
            return None
        return (ux, uy, align)
