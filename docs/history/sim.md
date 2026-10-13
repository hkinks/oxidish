> Historical: describes code removed from the repo; see [README.md](README.md).

# GPU agar.io simulator (`sim/`)

A batched PyTorch re-implementation of the MultiOgar arena we train against (`arena/server/arena.js`,
`vendor/MultiOgar/src`), built to feed PPO 10-100x more experience than the ~600 agent-steps/s the
real server gives. `B` independent arenas step in lockstep on the GPU; every player is an agent
(self-play is the default) and a vectorised scripted bot can drive some slots.

```
sim/engine.py       World: physics for B arenas (cells [B,P*16], food, viruses, ejected blobs)
sim/obs.py          Encoder: [15,48,48] grid (the guided net's 13 channels + killable + density) + 16 scalars
sim/tourney.py      head-to-head of nets in shared arenas; farm() = farming-alone check
sim/progress.py     learning curve of a run vs fixed reference checkpoints
sim/watch.py        live browser viewer (see sim-training.md)
sim/env.py          SimEnv: reset()/step(actions) -> grids, scalars, reward, done, valid (rl/env.py semantics)
sim/bots.py         scripted_actions(): flee threats / chase prey / eat food, vectorised
sim/test_sim.py     mechanics tests + encoder-vs-Python-encoder test (python -m sim.test_sim)
sim/compare.py      same scripted experiments on the REAL server and on the sim (python -m sim.compare)
sim/compare_obs.py  sim encoder vs rl/guided.py GuidedEncoder on live sim states
sim/bench.py        throughput (python -m sim.bench --B 128 256 512)
```

## Usage

```python
from sim.env import SimEnv
env = SimEnv(B=256, P=16)                       # 256 arenas x 16 players = 4096 agents, 7000x7000 map, 2x food
grids, scalars, reward, done, valid = env.reset()          # CUDA tensors, row = arena * n_learn + slot
grids, scalars, reward, done, valid = env.step(actions)    # actions [B*n_learn, 3] long: move 0..17, split, eject
```

* `grids [N,13,48,48]` (float32, `obs_dtype=torch.float16` halves it), `scalars [N,16]`, `reward/done/valid [N]`.
* `n_learn < P` makes the last slots scripted bots (`sim.bots`); only learner slots are encoded.
* `valid` is False while a slot is dead (its observation is zeros); the slot respawns on the next step,
  exactly like `ArenaEnv`: `done=True` on the death step, then one invalid step, then a new life.
* Reward = telescoped mass change / 200 (REWARD_MODE "linear"), death pays `-mass/200`, masses as the
  client sees them (floor(size)^2/100). Respawn curriculum: 50% of learner respawns start at a
  log-uniform mass in [50, 5000] (`curriculum_p`, `curriculum_mass`). Episodes truncate after
  `episode_steps` (default 1500 = 3000 ticks): `done=True` for living agents and the arenas restart.
* Moves 0..15 are headings (`rl/actions.py::decode`: target = centre + (cos, sin) * ext of the previous
  observation), 16 = stay; 17 (FOLLOW) also means stay, since the sim has no champion.
* The action is a mouse position: the world applies `int(x)` and the server's `~~(mouse - pos)`.

## Step semantics (one env step = 2 server ticks, sync=2)

1. respawn dead learners (spawn processed at message time, before the tick);
2. tick: `updateMerge`, `moveCell` (boost + border bounce/clamp), `movePlayer` (steer to the *old* mouse),
   `autoSplit`, collisions (push of unmerged own cells, eating, merging, virus pop, food, ejected), food/virus
   spawn every 20 ticks, decay every 25 ticks at `(tick+3) % 25 == 0`;
3. input phase: split, then eject (both aim with the **previous** step's mouse), then the mouse is
   updated (`PacketHandler.process` order: space, W, mouse). So a split fired with heading h goes along the
   *previous* heading, as on the real server;
4. second tick; the observation is the state after it (what the 0xC9 step packet precedes).

## Constants and where they come from

Config values written `mts(m)` are **sqrt(100 m) + 0.5** (`modules/ini.js:62-64` adds the 0.5), so the
effective sizes differ slightly from the comments in `config.ini`. `r2 = size^2`, `mass = r2 / 100`.

| quantity | value in sim | source |
|---|---|---|
| map, border | 7000 x 7000, centred on 0 | `arena.js:64`, `GameServer.setBorder` |
| food target | round(round(1500 (map/14142)^2) x mult) = 736 at 2x | `arena.js:68-71` |
| food spawn | min(target - count, 30) every 20 ticks at random spots | `GameServer.spawnCells` (L762), `config.ini` foodSpawnAmount/spawnInterval |
| food size/mass | size 10, mass 1; eaten when `d < size - 10/3` | `config.ini` foodMin/MaxSize, `resolveCollision` (L729) |
| virus min/max count | 12 / 25 (50 and 100 x area, min 4) | `arena.js:72-73` |
| virus spawn | 1 per 20 ticks if below the min and the spot is clear of cells/viruses (box test) | `spawnCells`, `willCollide` (L809) |
| virus size | 100.5 (mass 101), shoots a new virus at size >= 141.92 and goes back to 100.5 | `config.ini` virusMin/MaxSize, `Virus.onEat` |
| virus eat rule | cell size >= 1.1576 x virus, overlap `d < big - small/3`; one pop per player per tick | `config.ini` virusEatMult, `resolveCollision` |
| virus pop | `Virus.onEaten` port (`virus_splits`, tested against a literal port of the JS), pieces get boost 780 x size^0.0122 | `entity/Virus.js:29-60`, `GameServer.splitPlayerCell` (L653) |
| virus feeding | a blob is eaten by a virus only while virus count < 25; +blob r2 | `Virus.canEat/onEat` |
| speed | `2.2 * size^-0.439 * 40 * min(dist, 32) / 32` per tick, dist = `|~~(mouse - pos)|`, no move if dist^2 < 1 | `entity/PlayerCell.js:15`, `GameServer.movePlayer` (L625) |
| boost decay | each tick moves `bd/10` and `bd -= bd/10`; bounce off the border when the cell centre is within size/2 of it | `GameServer.moveCell` (L637) |
| border clamp | centre kept within `size/2` of the border (so cells poke half their radius outside) | `entity/Cell.js:70` |
| spawn size | 32.12 (mass 10.3); curriculum sizes `int(sqrt(100 mass))` | `config.ini` playerStartSize, `rl/env.py` |
| min split / eject size | 59.66 (mass 35.6); split needs size >, eject size >= | `config.ini` playerMinSplit/MinEject, `splitCells` (L820), `ejectMass` (L847) |
| split | parent and child get half the area (`size/sqrt2`), child boost 780 x size^0.0122 along `atan2(mouse - cell)`; zero aim vector -> +y | `splitPlayerCell`, `splitCells` |
| max cells | 16 | `config.ini` playerMaxCells |
| merge | `canRemerge` when age >= max(30 s, 0.2 x size) x 25 ticks; age counts from a cell's own birth (the parent keeps its age) | `updateMerge` (L590) |
| own cells before merge | pushed apart (full overlap per pair, split by mass ratio) once both are >= 13 ticks old and not both remergeable; no interaction before 13 ticks | `checkRigidCollision` (L687), `resolveRigidCollision` (L712), splitRestoreTicks |
| merge | both remergeable: the larger eats the smaller when `d < big - small/3`, no size-ratio test | `resolveCollision` |
| eat other players | size >= 1.15 x prey and `d < big - small/3` | `resolveCollision`, config playerEatMult |
| big cells ignore crumbs | a cell of size >= 250 gains nothing from a player cell of size <= 41.23 | `entity/Cell.js:53` (playerBotGrow = 0) |
| decay | every 25 ticks (at `(tick+3)%25==0`) r2 *= 0.998 per cell, floor at size 32.12; the loop `break`s at the first cell at the floor | `updateDecay` (L597), `config.ini` playerDecayRate |
| auto split | size >= 1500.5: halve at a random angle, or clamp at 16 cells | `autoSplit` (L615) |
| eject | size >= 59.66: cell loses r2 of size 41.73 (mass 17.4); blob size 36.56 (mass 13.4) spawned at `pos + dir*size`, angle jitter +-0.3 rad, boost 780; blobs cannot be eaten at age 0 | `ejectMass`, `config.ini` eject*, `resolveCollision` |
| eject cooldown | 2 ticks - never binding at one eject per 2-tick step | `canEject` (L838) |
| blob vs blob | rigid push (ejectCollisionType 0) | `mainLoop` nodesMoving loop |
| respawn | random spot (<= 10 re-rolls to avoid cells/viruses) at the spawn size | `spawnPlayer` (L782) |
| view box | centre = unweighted mean of own cells, half-width 960/s, half-height 540/s, `s = max(min(64/sum sizes, 1)^0.4, 0.15)`; objects whose bounding box intersects it are sent | `PlayerTracker.updateTick`, `updateScale` |

## Observation encoder

`sim/obs.py` reproduces `rl/obs.py::encode(G=48, server_view=True)` plus `rl/guided.py::GuidedEncoder`:
channel order food, virus (fullness 0.3..1), mine, prey, threat, split_threat, ejected, outside, vx, vy,
own_merge, enemy_merge, owner_mass; scalars `[log(total)/8, n_cells/16, main/total, cx, cy, ext/3000, main>=36,
log(main)/8]` followed by 8 hint scalars. What it mimics because the real encoder sees it:

* sizes are floored (uint16 in the packet) and positions truncated before use;
* only objects inside the server view box exist, so the 48x48 grid (centred on the mass-weighted centroid,
  half-extent `960 / min(64/sum sizes,1)^0.4`) is empty beyond +-540/s vertically, and food/players near the
  edge of the box are cut exactly as in the real client;
* velocity is measured over up to 3 steps by cell id (history reset at death), merge readiness uses ids and the
  first step each agent saw a cell (`first_seen`), owner mass groups the *visible* pieces of a player;
* overwrite channels (vx, vy, owner, virus) resolve overlaps with the newest cell winning, own cells lowest.

**The 8 hint scalars (champion direction, distance, split, eject, farm/hunt/flee) are zeros** - there is no
champion in the sim. A net trained here with the hint zeroed has to be fine-tuned on the real server (or the hint
channels masked) to use them.

## Known deviations / simplifications

* **Simultaneous collision resolution.** The server resolves cell by cell in list order; the sim resolves all
  pairs of a tick at once: a prey with several eaters goes to the largest, an eater that is itself eaten this tick
  eats nothing (its prey survives to the next tick), at most one virus pop per player per tick, a blob goes to a cell
  before a virus, a pellet to the first covering slot, rigid pushes are summed instead of applied sequentially.
* **Slot order stands in for creation order** (`client.cells`): matters only for which cells split when more than
  16 would exist and for the decay `break`.
* **Capacity limits.** 16 cell slots per player; ejected blobs: 256 per arena, the oldest is recycled when full
  (the real server never removes a blob; `world.eject_overwrites` counts recycled ones); viruses 29 slots.
* **Boost residue.** Boost below 0.05 units is dropped (the server keeps a ~0 residue for thousands of ticks, with
  collisions active; I treat all blobs/viruses as always collision-active, which matches that).
* **Several blob feeds of one virus in the same tick** produce one shot (sequentially there could be more).
* **Spawn-from-blob.** `ejectSpawnChance` (1% with the .5 config) lets a respawning player take over a stopped blob;
  not modelled.
* **World starts full.** Real arenas start with no food/viruses and fill over ~500 ticks; `World(prefill=True)`
  (default) starts at the steady state. Use `prefill=False` for the real behaviour. An arena that is reset
  (episode end) is observed at tick 0, the real env after 2 ticks.
* **No server bots, no mother cells, no leaderboard, no spectators**; FFA only. Arenas truncate together.
* **float32 state** (server: float64); positions near the border resolve to 2e-4, irrelevant but not bit-exact.
* **Encoder float edge cases.** `GuidedEncoder` runs in float64 on the CPU; the sim does the frame / grid transform in
  float64 too, but a CUDA `pow` can differ by an ulp and a pellet or disc edge landing exactly on a pixel boundary then
  flips. Measured: at most 0.1% of agent-observations differ in any pixel of any channel (below).

## Verification (what is checked against what)

Run: `python -m sim.test_sim` (15 tests, each mechanics test runs eagerly and under CUDA graphs),
`python -m sim.compare` (needs node; starts its own arena servers on free ports and terminates them),
`python -m sim.compare_obs`, `python -m sim.bench`.

**1. Mechanics vs the real server** (`sim/compare.py`; identical scripted decision logic on a real arena
server, no bots, 1 food pellet, and on the sim started from the same cells; measured from what the client sees).
Last run, full output in `logs/sim_compare.txt`:

| experiment | quantity | real server | simulator |
|---|---|---|---|
| speed, size 32 / 60 / 100 / 200 / 400 / 700 | units per step | 38.552 / 29.292 / 23.287 / 17.279 / 12.780 / 10.037 | 38.552 / 29.292 / 23.303 / 17.279 / 12.780 / 10.037 |
| split, size 100 | cell separation 1 / 2 / 5 / 10 / 25 steps after | 82.098 / 222.380 / 503.779 / 711.126 / 817.240 | identical to 3 decimals |
| split, size 400 | same | 84.149 / 226.374 / 511.822 / 722.222 / 831.387 | identical to 3 decimals |
| split pieces | masses | [49, 49], [795, 795] | same |
| merge, size 100 | steps from split to merge (formula 375) | 378 | 378 |
| merge, size 400 | steps from split to merge (formula 707) | 692 | 693 |
| decay, size 316 | % mass lost per second | 0.202 | 0.202 |
| eat (A size 100 holds at distance d from B, size 86, ratio 1.163) | d = 66 / 70 / 74 / 78 | eaten at step 132 (mass 169) / survives / survives / survives | same, same step |
| eat (size 88, ratio 1.136, d = 60) | outcome | survives | survives |
| eat (size 86, d = 60) | outcome | eaten at step 131, mass 169 | same |
| eject, mass 100 (client mass after) | mass | 81.000 | 81.000 |
| eject blob travel 1 / 3 / 8 / 15 steps after first sight | units | 108.5 / 266.4 / 463.1 / 544.1 | 108.2 / 266.4 / 463.6 / 544.7 |
| virus pop, mass 480 cell | pieces / mass / biggest sizes | 16, 573, [164, 45, 45, 45, 45, 45] | 16, 573, [164, 45, 45, 45, 45, 45] |
| view box (1689-4356 objects, 3 sizes) | max (offset - size) / half-extent, x / y | 1.000 / 0.998, 1.001 / 1.001, 0.999 / 1.000 | assumed <= 1 |

Residual differences, not investigated further: sim speed at size 100 is 0.07% higher, the size-400 merge takes one
step longer in the sim, and the blob travel differs by up to 0.5 units. The real blob direction is random (+-0.3
rad) and sizes/positions are integers on the client, so sub-unit differences of this kind are plausible noise, but I
have not shown that. Eject mass was off by 1.8 in the first run, which exposed that `mts(m)` is `sqrt(100 m) + 0.5`
(see the constants table); fixed.

**Not checked against the real server** (verified only against my reading of the JS, and by the unit tests):
virus feeding / shooting a new virus, ejected-blob pushing, food and virus respawn dynamics, cell-cell
interaction among many players (simultaneous resolution is a deliberate approximation), respawn placement,
split chains beyond one split, the big-cell/crumb rule, auto split at 1500, and border behaviour of boosted cells
(unit-tested against the formulas only).

**2. Observation encoder vs `rl/guided.py`** (`sim/compare_obs.py`: a `Snapshot` is built from sim state exactly as
the client sees it and fed to the real `GuidedEncoder` with its history; 4 seeds x 100 steps x 2 arenas x 8
players, scripted bots plus random splits/ejects, curriculum spawns up to mass 3000). Fraction of agent-observations
with any pixel differing by more than 1e-3: food 0.0002, ejected 0.0002, vx 0.0011, vy 0.0013, every other channel
and all base scalars 0.0000. (This also caught a real bug: `prey`/`threat` values need the `min(1, ...)` clamp.)

**3. Mechanics unit tests** (`sim/test_sim.py`): speed vs size, steering near the mouse (with `~~` truncation),
eat ratio and overlap, food, decay, split distance/halving, split uses the previous mouse, merge timer and the push
of unmerged cells, virus pop, virus too small to eat, eject, virus feed -> shoot, border clamp, `virus_splits` against a
literal port of `Virus.onEaten` (500 random cases), random-world invariants (finite, <= 16 cells, mass floor),
encoder vs Python encoder.

## Performance (`python -m sim.bench`, RTX 5090 shared with two PPO trainings and ~10 arena servers)

Agent-steps/s including the observation encoder (random actions, P = 16; GPU memory peak 1.1 GB at B = 256):

| B | agents | with obs | physics only |
|---|---|---|---|
| 64 | 1024 | 19-33k | 70-170k |
| 128 | 2048 | 33-71k | 230-320k |
| 256 | 4096 | 70-138k | 390-630k |
| 512 | 8192 | 115-237k | 700-1100k |

The ranges are several runs on a machine whose CPU is busy: the sim is launch/sync bound (physics runs as CUDA graphs,
the encoder's static half too; 4 host syncs per step remain: counts after each tick, visible-object count, op counts)
so throughput is roughly independent of B until B ~ 500 and depends on how loaded the CPU / GPU time-slicing is.
Use B >= 128-256 to be above 50k. The grids tensor is large (13x48x48 floats = 120 KB per agent, 490 MB per step at
B = 256): use `obs_dtype=torch.float16` or consume it immediately.


## Later additions (2026-10-04)

* **Kills:** `World.kstat [2,B,P+1]` counts enemy cells eaten and enemy mass eaten per player, credited to the eater's owner. `SimEnv(kill_reward=(cell, mass))` rewards them, and per-life `kills` shows up in `pop_stats()`.
* **Inactive slots:** `SimEnv.active [B,P]` slots never (re)spawn. The trainer's farming drills use this.
* **Encoder:** two more channels. `killable` marks enemy cells under 1/2.6 of our largest cell. `density` is the log of enemy mass summed per pixel, so overlapping pieces add up instead of max-blending into one blob.
* Training on the simulator, the evolution loop and the viewer are described in [sim-training.md](sim-training.md).
