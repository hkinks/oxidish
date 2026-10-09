# Sparring opponents: AposBot port and hunters (2026-10-07)

Why: every opponent in the field (champ, r2g9, safe, ramp_r2, ramp300) is the same potential-field rule bot with
different genes: cautious farmers that rarely hunt and never use viruses. c3_4 is evolved against them, so they
give little pressure. This adds opponents that hunt and use viruses.

## What there is

| Name | Spec | What |
|---|---|---|
| `apos` | `{"kind": "apos"}` (`bot/sparring/apos.json`, or the lineup key `apos`) | native port of Apostolique's AposBot decision logic |
| `hunter_hunt` | `bot/sparring/hunter_hunt.json` | c3_4 farming genes + chase everything from 1800 away, split-kill, double split |
| `hunter_virus` | `bot/sparring/hunter_virus.json` | + stages and feeds viruses at big players, hides behind viruses |
| `hunter_reckless` | `bot/sparring/hunter_reckless.json` | + chases everything (no prey floor), up to 4 pieces |

Use them like any genome: `arena.tournament --engine ogar --agents apos=bot/sparring/apos.json:1
hhunt=bot/sparring/hunter_hunt.json:1 ...`, `arena.endless --agents ...`, `ogar match|serve --agents ...`.
`arena.make_sparring` regenerates the hunter files.

## AposBot port (`bot/apos.py`, `rust/arena_client/src/apos/`)

Port of module 9 of `vendor/apos/aposbot.user.js` (`createAposBot(api)`): the 64-heading scoring (threats with
split reach, cover, flight lines, walls, food kernel, prey with catchability / traps, viruses, escape room),
heading hysteresis, target commit, flee/bait/gather strides, `planPop` (stage round a virus and feed it at a big
player), `planEscape` (split out of a pin), `planLure`, `planHide`, `planPour`, `planSplit`. Constants and branch
order follow the JS. Spec options: `aggressive` (the JS `E` toggle, default true), `step_ms` (default 80 = sync 2),
`memory_ms` (3000).

Deviations (forced by what a `Snapshot` carries; also in the module docstring):
- Time: JS reads the wall clock; here `now` advances `step_ms` per decision (starts large, like an epoch clock).
- Cell colour (grouping a player's pieces, ejected mass) is the protocol colour, new `Cell.color` (Python client,
  Rust client and the ogar in-process snapshot fill it). Names are not used: only `kind == food` is a pellet;
  `ejected` is `kind == ejected` (the headless JS shim never sets it).
- `getMemoryCells()` is rebuilt from past snapshots (players and viruses, 3 s); a remembered cell that should be in
  view but is missing is dropped (eaten).
- `verticalDistance()` = view-box diagonal 1920x1080 / scale (scale = (64/sum of radii)^0.4, min 0.15).
- `Math.hypot` -> `sqrt(dx*dx+dy*dy)`, `Math.round` -> `floor(x+.5)`; own cells sorted by size then id, other
  cells by id (JS integer-key order); drawing dropped.
- Splits and ejects are flags of the decision and fire toward the returned aim point (JS fires on the previous
  pointer, hence its alignment checks, kept).

Parity: `arena.rust_tools dump` records real lockstep matches (node server) with Python decisions;
`arena_client parity` replays them in Rust. Dump format now carries cell colour. Results below.

## Hunter genes (`bot/tactics.py`, `rust/arena_client/src/policy.rs`, all default off, in Python and Rust)

- `shoot_stage` > 0.5: with the shoot skill on and no shot lined up, walk round a virus (within `shoot_walk`) to the
  side away from a big player, `shoot_standoff` from its edge (the AposBot stage), and hold position while
  shooting (aim just past our edge instead of at the virus).
- `shoot_safety` (0.7): shot called off while a threat is within this x its danger radius.
- `shoot_max_cells` (1): shoot while we have at most this many cells.
Not in `BOUNDS`, so evolution is unchanged; existing genomes behave identically (parity re-run on c3_4 / champ).

## Results

**Parity (Python vs Rust, `arena.rust_tools dump` on real node-server matches + `arena_client parity`):**
26996 decisions (3 apos, hunter_virus, c3_4, champ), 0 aim mismatches, 0 split/eject mismatches, max |aim diff| 5e-12;
an earlier 26996-decision dump (4 apos + 2 route bots) and 16000 fuzz decisions (hunter genes with staging, c3_4, champ)
also 0 mismatches. The apos runs exercised flee, bait, lure, hide, pour, escape-split, hunt-split, stage and pop.

**15-min matches** (`arena.tournament --engine ogar --matches 24 --ticks 22500 --food-mult 2`, one copy each, 8 server bots, seeds 5000+):

| bot | log mass | avg mass | deaths | kills (cells) | virus shots | virus pops |
|---|---|---|---|---|---|---|
| c3_4 | 6.95 | 1339 | 3.0 | 62 | 0 | 0 |
| ramp300 | 6.89 | 1199 | 4.0 | 61 | 0 | 0 |
| rampr2 | 6.76 | 1184 | 4.3 | 63 | 0 | 0 |
| champ | 6.71 | 1133 | 4.7 | 45 | 0 | 0 |
| r2g9 | 6.69 | 1058 | 4.8 | 69 | 0 | 0 |
| safe | 6.58 | 1061 | 3.8 | 41 | 0 | 0 |
| hunter_hunt | 6.53 | 946 | 4.8 | 98 | 0 | 0 |
| hunter_virus | 5.94 | 452 | 6.1 | 87 | 1.8 | 1.5 |
| apos | 5.87 | 421 | 4.1 | 61 | 0.3 | 0.3 |
| hunter_reckless | 5.75 | 356 | 8.5 | 89 | 1.4 | 1.3 |

c3_4 still beats every one of them head to head (71-96% of matches vs the hunters/apos). hunter_hunt is the top
killer of the other bots (13-18% of c3_4's, champ's, ramp's lost mass), apos kills a lot of pieces but dies in
splits (76% of its lost mass while split). Virus shots are rare (a staged feed needs ~8 aligned ejects while a
big player stays put): ~2 per hunter_virus per 15 min, nearly every shot popped someone.

**Endless worlds** (`arena.endless`, 4 worlds x 240 game-min, windows of 10 min, first 30 min skipped, seeds 800+;
field = c3_4 x2, champ, r2g9, safe, rampr2, ramp300; "+ sparring" adds one each of apos, hunter_hunt,
hunter_virus, hunter_reckless):

| | c3_4 leader share | c3_4 dominance (heaviest family) | c3_4 mean mass |
|---|---|---|---|
| field only | 32% | 22% (19/88) | 3411 |
| + sparring | 38% | 31% (27/88) | 3404 |

Family mean mass with sparring: c3_4 3404, ramp300 2770, champ 2463, rampr2 2122, r2g9 2038, safe 1783, hunter_hunt 1202,
apos 750, hunter_virus 471, hunter_reckless 424 (leader shares: c3_4 38%, ramp300 16%, champ 14%, r2g9 12%, rampr2 8%,
safe 7%, hunter_hunt 2%, apos 2%). So the sparring bots do not lower c3_4 (its mass is unchanged); they take mass
from the other farmers, which is why its leader share goes up. They are pressure on everybody, not a stronger
bot to beat: none is heavy enough to lead. A hunter that is also a good farmer would be the next step.

**Native vs real AposBot** (`arena.apos_compare --minutes 12`, real time, one world: 2 native, 2 real JS bots in
headless Chromium, champ x1): mean mass per bot native 300 / 524, real 165 / 1566; family means native 412
(peak 2199), real 866 (peak 1982), champ 642; nobody died in the 12 min. One run, 144 samples, 2 bots per side:
the per-bot spread (165 to 1566) is larger than the gap, so it only says "same league", not "same strength".
