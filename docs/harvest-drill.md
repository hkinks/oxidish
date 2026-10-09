# Giant-pop harvest drill

How well does an agent eat a giant that has just burst on a virus? The user's claim: "a good 500 mass hunter can
completely overtake a 5k mass giant" after it pops. Until now we had only a coarse counter (`harvest_mass` per hour in
`ogar endless`). The drill is a direct, repeatable, seeded measurement: `ogar drill` (`rust/ogar/src/drill.rs`), tables
from `arena/drill_report.py`.

Short answer (numbers below, N = 100 reps per cell for the headline, 50 for the sweeps, 95% intervals):

* The arithmetic allows it. A 5k giant that pops once leaves 12 pieces of 28 mass, one of 318, 637, 1275 and a 2521 core. A 500 hunter
  that eats them smallest first reaches 500 -> 1154 -> 1791 -> 3066 and is then heavier than the 2521 core: the
  **mass-chain bound is 51% of the giant's mass and an overtake**. The 12 small pieces plus the 318 are the "easy" 13% of the giant.
* In play nobody gets near that bound. Within 30 s of the pop the best agents capture **13-17%** of the giant
  (the easy class and a bit), overtake it in **5-12%** of the reps (28-44% when given 60 s) and die in 7-12% (12-34% within 60 s).
  The giant (a c3_4 rule bot) hunts the hunter with its big pieces, which are as fast as the hunter.
* Learned and rule agents are indistinguishable when the pop happens in front of them (`giant_popped`: c3_4 13%, p3 14%,
  p6 11%, the scripted oracle 17%, all intervals overlap). They are 3-15x worse than the oracle when the giant walks onto
  the virus while they are live (`giant_pop`: rule bots 4-5%, p3 4%, p6 1%, oracle 15%), because they have moved away
  from the pop by the time it happens and then do not go back for the pieces.

## Running it

    # from the repo root (models are gitignored: copy runs/vision/export/*.onnx and runs/vision/ort from the main checkout)
    cargo build --release -j 2 --manifest-path rust/ogar/Cargo.toml        # use your own CARGO_TARGET_DIR
    ogar drill --agents c3_4=bot/c3_4.json harv=bot/drill/c34_harvest_r25.json p3=bot/vision_p3.json oracle=oracle \
               --scenario giant_pop,giant_popped --our-mass 500 --giant-mass 5000 --dist 600 --giant-mode bot \
               --reps 100 --seed 1 --threads 4 --out runs/drill/x.json
    python -m arena.drill_report runs/drill/x.json [--what main|diag|compact]

`--our-mass`, `--giant-mass`, `--dist`, `--scenario` and `--giant-mode` take comma lists (a cell per combination).
Other options: `--competitor spec.json --comp-mass 800`, `--giant-bot spec.json` (default `bot/c3_4.json`), `--horizon 30`
(seconds after the pop; windows 10/20/30 and the horizon if longer), `--map`, `--food-mult` (default 2, as the endless evaluator).
Agent specs: any rule bot / planner / vision json (`name=file.json`), or `oracle`, `oracle_near`, `oracle_nosplit`, `oracle_view`, `oracle_brave`.
Rep `i` of every agent and cell uses seed `seed + i` (paired comparison). Vision nets need a CUDA setup
(`OGAR_DLL_DIRS` = a dir with cuDNN 9, e.g. the torch wheel's `lib`) or `OGAR_VISION_EP=cpu` (~10x slower). The "cuda" graph backend died once under
6 threads (cuDNN error, not reproduced); `cuda_plain` and `cpu` are slower but safe. `OGAR_DRILL_TRACE=1` prints a line per second per rep.
Tests: `cargo test --release drill` (scene determinism per seed, geometry, oracle rep determinism, interval maths).

## The scene

A world from a seed: map 7000, 2x food, **no server bots and no stray viruses** (the one virus of the scene is the only one).
Our agent (`us`), the giant, optionally a third-party rule bot (`comp`).

* Geometry: the virus V sits near the map centre; we stand `D` (`--dist`) from it. The giant bursts on it from the far side, its centre
  `giant size - 34` beyond V (just inside its eating range), so centre-to-centre distance at the pop is `D + giant size - 34`
  (D = 600, 5k giant: 1273) and the gap between the two edges is `D - 34 - our size` (343). The competitor stands `D` from V, 55-120 degrees round from us.
* `giant_popped`: the giant starts on the virus and pops on the first tick; our agent is live from tick 0. The clock starts at the pop.
* `giant_pop`: the giant starts 500 further out and walks onto the virus (about 4 s) while our agent plays normally. The clock starts at the pop. We report the real centre
  distance at the pop (`dist_at_pop`): an agent that backs away from the approaching giant is farther.
* The pop is the real `virus_eaten` code (MultiOgar rules, 16 cells max), so the piece distribution is the game's: giant 5000 -> 12 x 28, 318, 637, 1275, 2521 mass (all within ~1 s the pieces fly up to ~840 away); 3000 -> same shape, core 1500.
* Giant behaviour after the pop (`--giant-mode`): `bot` (default, the realistic one): a rule bot (c3_4) drives the pieces, so it hunts, splits, flees, uses viruses.
  `hunt`: one cursor, on our nearest cell when a piece that can eat it is within 1100, else on the pieces' centroid. `passive`: the cursor stays on the centroid (the pieces regroup within a second).
  Before the pop a scripted walker steers it onto the virus in every mode.
* Competitor: a c3_4 rule bot with `--comp-mass` (800) mass.

## Metrics (per rep, from the pop on)

* `capture`: mass of giant pieces **we** ate / the giant's total mass right after the pop, cumulative at 10/20/30 s (and the horizon). Mass eaten by the competitor is `stolen_by_comp`.
* `ours/giant`: our total mass over the giant's remaining total at the window (capped at 5).
* `overtake`: our mass is above the giant's remaining total at any tick (rate and, for the reps that do, the median time); `overtake_any` is within the horizon, the per-window rate is in the JSON.
* `first piece`: rate of reps that eat at least one piece, and the mean time to the first (over those that did); `pieces`: giant pieces eaten.
* `died` (and by whom: giant / third party), `lost_third`: mass of ours eaten by anybody but the giant.
* `chain_cap`/`chain_overtake`: the mass-chain bound above (smallest piece first, no chasing, no danger, no food).
* Behaviour counters sampled every decision (diagnosis tables): share of ticks with an edible giant piece in the view box; of those, the share the agent moves towards the nearest one (cosine > 0.5) or stands still; share of ticks with a piece that can eat it within 400; of those, the share it moves away; splits per rep; max cells; peak mass.
* Reps where the agent dies before the pop or the giant never pops are dropped from the cell (column `n`; practically only `oracle_brave` loses reps: 3-6 per 50-100).

Intervals: 95%; Wilson for rates, mean +- 1.96 se for means. N = 100 gives +-8pp on a 50% rate, +-2-3pp on a 15% capture.

## Agents

* Rule bots: `c3_4` (`bot/c3_4.json`), `plan` (`bot/plan_c34.json`, the lookahead planner on c3_4), `harv` (c3_4 + `harvest = 1`), `harv_r25` / `harv_r4` (+ `harvest_range` 2.5 / 4), `plan_harv` / `plan_harv_r25`
  (planner on those). Variants are in `bot/drill/`.
* Vision nets `p6` (`bot/vision_p6.json`) and `p3` (`bot/vision_p3.json`), split off, CUDA.
* **Scripted harvesters** (not an upper bound, see below): `oracle` sees every piece on the map, scores a piece by mass / (edge distance + 100) like the rule bots do, flees along the clearest of 16 directions when a piece that can eat it is within 200 (500 if its half could eat us too), only approaches pieces with no such threat within size + 100 and not on its path, splits onto a piece when its half cell would still eat it and no threat is near (almost never), eats food otherwise.
  `oracle_view` only sees its view box; `oracle_near` takes the nearest piece (the user's rule); `oracle_nosplit`; `oracle_brave` ignores all danger (to see what fear costs).
* The vision-reflex variants of the other agent (vision-m5) are **not on master yet**, so they are not measured. Drop their specs into `--agents` when they land.

## Results

Headline: 500 vs a 5k giant, D = 600, c3_4 giant, 30 s. Columns: capture at 10/20/30 s, ours/giant at 30 s, overtake at any time within 30 s, median time of those, first-piece rate / time, died.

### `giant_popped` (the pop happens in front of the agent), N = 100

**giant_popped/our500/giant5000/d600/bot**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 100 | 8+-1% | 10+-2% | 13+-2% | 0.29+-0.05 | 6% [3%-12%] | 26.2 s | 95% / 1.8 s | 7% [3%-14%] | 9.8 |
| plan | 100 | 7+-1% | 9+-2% | 11+-2% | 0.27+-0.05 | 7% [3%-14%] | 26.2 s | 93% / 1.8 s | 7% [3%-14%] | 9.3 |
| harv | 100 | 8+-1% | 10+-2% | 13+-3% | 0.30+-0.06 | 8% [4%-15%] | 26.2 s | 95% / 1.8 s | 8% [4%-15%] | 9.7 |
| harv_r25 | 100 | 8+-1% | 11+-2% | 14+-3% | 0.31+-0.06 | 8% [4%-15%] | 26.2 s | 95% / 1.8 s | 9% [5%-16%] | 10.0 |
| harv_r4 | 100 | 8+-1% | 11+-2% | 14+-3% | 0.31+-0.06 | 8% [4%-15%] | 26.2 s | 95% / 1.8 s | 9% [5%-16%] | 10.0 |
| plan_harv | 100 | 6+-1% | 8+-1% | 9+-2% | 0.23+-0.04 | 3% [1%-8%] | 22.6 s | 93% / 1.8 s | 8% [4%-15%] | 8.4 |
| plan_harv_r25 | 100 | 6+-1% | 8+-1% | 10+-2% | 0.23+-0.05 | 4% [2%-10%] | 26.0 s | 93% / 1.8 s | 9% [5%-16%] | 8.4 |
| oracle | 100 | 8+-1% | 12+-2% | 17+-3% | 0.31+-0.05 | 10% [6%-17%] | 20.7 s | 96% / 1.5 s | 12% [7%-20%] | 11.3 |
| oracle_near | 100 | 8+-1% | 12+-2% | 17+-3% | 0.36+-0.07 | 12% [7%-20%] | 21.8 s | 96% / 1.5 s | 12% [7%-20%] | 11.3 |
| oracle_view | 100 | 8+-1% | 12+-2% | 17+-3% | 0.35+-0.06 | 11% [6%-19%] | 21.6 s | 96% / 1.5 s | 12% [7%-20%] | 11.2 |
| oracle_brave | 100 | 4+-1% | 5+-1% | 8+-2% | 0.04+-0.04 | 2% [1%-7%] | 29.7 s | 90% / 0.8 s | 91% [84%-95%] | 5.4 |
| p6 | 100 | 5+-1% | 8+-1% | 11+-2% | 0.26+-0.05 | 5% [2%-11%] | 23.2 s | 95% / 1.9 s | 7% [3%-14%] | 9.7 |
| p3 | 100 | 6+-1% | 12+-2% | 14+-2% | 0.31+-0.06 | 7% [3%-14%] | 18.1 s | 96% / 1.8 s | 10% [6%-17%] | 10.5 |

### `giant_pop` (the giant walks onto the virus while the agent plays), N = 100

**giant_pop/our500/giant5000/d600/bot**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 100 | 3+-1% | 3+-1% | 5+-2% | 0.16+-0.02 | 0% [0%-4%] | - s | 31% / 8.2 s | 3% [1%-8%] | 3.4 |
| plan | 100 | 2+-1% | 2+-1% | 4+-2% | 0.17+-0.03 | 1% [0%-5%] | 23.5 s | 28% / 8.7 s | 0% [0%-4%] | 2.9 |
| harv | 100 | 2+-1% | 3+-1% | 4+-2% | 0.15+-0.02 | 0% [0%-4%] | - s | 28% / 9.4 s | 3% [1%-8%] | 2.9 |
| harv_r25 | 100 | 2+-1% | 3+-1% | 5+-2% | 0.16+-0.02 | 0% [0%-4%] | - s | 32% / 10.3 s | 3% [1%-8%] | 3.4 |
| harv_r4 | 100 | 2+-1% | 3+-1% | 5+-2% | 0.16+-0.02 | 0% [0%-4%] | - s | 32% / 10.3 s | 3% [1%-8%] | 3.4 |
| plan_harv | 100 | 1+-1% | 2+-1% | 3+-1% | 0.14+-0.02 | 0% [0%-4%] | - s | 27% / 9.6 s | 0% [0%-4%] | 2.3 |
| plan_harv_r25 | 100 | 2+-1% | 2+-1% | 4+-1% | 0.15+-0.02 | 0% [0%-4%] | - s | 30% / 9.5 s | 1% [0%-5%] | 2.9 |
| oracle | 100 | 6+-1% | 10+-2% | 15+-3% | 0.30+-0.06 | 7% [3%-14%] | 24.5 s | 93% / 2.8 s | 9% [5%-16%] | 10.4 |
| oracle_near | 100 | 6+-1% | 10+-2% | 14+-2% | 0.30+-0.06 | 6% [3%-12%] | 25.0 s | 93% / 2.8 s | 9% [5%-16%] | 10.4 |
| oracle_view | 100 | 6+-1% | 10+-2% | 14+-3% | 0.30+-0.06 | 7% [3%-14%] | 24.7 s | 90% / 3.1 s | 7% [3%-14%] | 9.7 |
| oracle_brave | 94 | 5+-1% | 6+-1% | 7+-2% | 0.04+-0.03 | 1% [0%-6%] | 23.2 s | 85% / 1.4 s | 87% [79%-93%] | 5.7 |
| p6 | 100 | 0+-0% | 1+-1% | 1+-1% | 0.12+-0.01 | 0% [0%-4%] | - s | 13% / 13.9 s | 2% [1%-7%] | 1.1 |
| p3 | 100 | 1+-0% | 3+-1% | 4+-2% | 0.18+-0.04 | 3% [1%-8%] | 25.0 s | 22% / 10.1 s | 0% [0%-4%] | 2.4 |

(`oracle_brave` n = 94: it is eaten before the pop in 6 reps.)

### 60 s horizon, N = 50

**giant_popped/our500/giant5000/d600/bot**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 8+-2% | 11+-3% | 12+-3% | 0.28+-0.08 | 28% [17%-42%] | 45.3 s | 94% / 1.9 s | 14% [7%-26%] | 10.3 |
| plan | 50 | 7+-2% | 9+-3% | 11+-4% | 0.27+-0.08 | 22% [13%-35%] | 45.2 s | 90% / 1.9 s | 14% [7%-26%] | 9.5 |
| harv_r25 | 50 | 7+-2% | 11+-3% | 14+-4% | 0.32+-0.09 | 34% [22%-48%] | 40.1 s | 94% / 1.9 s | 12% [6%-24%] | 10.7 |
| oracle | 50 | 7+-2% | 13+-3% | 17+-4% | 0.35+-0.09 | 32% [21%-46%] | 41.4 s | 96% / 1.6 s | 34% [22%-48%] | 11.9 |
| oracle_view | 50 | 7+-2% | 13+-3% | 17+-4% | 0.37+-0.10 | 36% [24%-50%] | 40.9 s | 96% / 1.6 s | 32% [21%-46%] | 12.1 |
| p6 | 50 | 5+-1% | 7+-2% | 10+-3% | 0.24+-0.06 | 32% [21%-46%] | 42.5 s | 94% / 1.9 s | 18% [10%-31%] | 10.6 |
| p3 | 50 | 6+-2% | 12+-3% | 15+-4% | 0.32+-0.08 | 44% [31%-58%] | 42.4 s | 96% / 1.9 s | 14% [7%-26%] | 11.4 |

| agent | capture 60s |
|---|---|
| c3_4 | 20+-5%, overtake 28% |
| plan | 16+-5%, overtake 22% |
| harv_r25 | 23+-5%, overtake 34% |
| oracle | 26+-5%, overtake 32% |
| oracle_view | 27+-5%, overtake 36% |
| p6 | 22+-5%, overtake 32% |
| p3 | 27+-6%, overtake 44% |

**giant_pop/our500/giant5000/d600/bot**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 3+-2% | 3+-2% | 5+-2% | 0.17+-0.03 | 12% [6%-24%] | 49.7 s | 44% / 13.6 s | 12% [6%-24%] | 4.4 |
| plan | 50 | 2+-1% | 3+-1% | 4+-2% | 0.16+-0.03 | 4% [1%-13%] | 58.9 s | 44% / 13.6 s | 10% [4%-21%] | 3.8 |
| harv_r25 | 50 | 3+-1% | 3+-2% | 5+-2% | 0.18+-0.04 | 10% [4%-21%] | 44.7 s | 46% / 15.5 s | 8% [3%-19%] | 4.7 |
| oracle | 50 | 5+-1% | 10+-2% | 14+-3% | 0.30+-0.07 | 42% [29%-56%] | 42.0 s | 94% / 4.6 s | 28% [17%-42%] | 11.0 |
| oracle_view | 50 | 5+-1% | 9+-2% | 13+-4% | 0.29+-0.08 | 34% [22%-48%] | 42.0 s | 90% / 4.0 s | 24% [14%-37%] | 10.1 |
| p6 | 50 | 0+-0% | 1+-1% | 2+-1% | 0.13+-0.02 | 8% [3%-19%] | 44.2 s | 24% / 26.2 s | 6% [2%-16%] | 1.9 |
| p3 | 50 | 1+-1% | 3+-2% | 4+-3% | 0.18+-0.06 | 14% [7%-26%] | 38.6 s | 26% / 16.8 s | 2% [0%-10%] | 3.0 |

| agent | capture 60s |
|---|---|
| c3_4 | 11+-5%, overtake 12% |
| plan | 8+-3%, overtake 4% |
| harv_r25 | 10+-4%, overtake 10% |
| oracle | 26+-6%, overtake 42% |
| oracle_view | 23+-6%, overtake 34% |
| p6 | 5+-4%, overtake 8% |
| p3 | 9+-5%, overtake 14% |

### Mass ratio and distance (`giant_popped`, c3_4 giant, N = 50; cells: capture at 30 s / overtake / died)

| condition | c3_4 | plan | harv_r25 | oracle | oracle_view | p6 | p3 |
|---|---|---|---|---|---|---|---|
| giant_popped/our300/giant5000/d600/bot | 0+-0% / 0% / 0% | 0+-0% / 0% / 0% | 1+-1% / 0% / 0% | 1+-1% / 0% / 2% | 1+-1% / 0% / 2% | 0+-0% / 0% / 0% | 0+-0% / 0% / 0% |
| giant_popped/our500/giant5000/d600/bot | 13+-2% / 6% / 7% | 11+-2% / 7% / 7% | 14+-3% / 8% / 9% | 17+-3% / 10% / 12% | 17+-3% / 11% / 12% | 11+-2% / 5% / 7% | 14+-2% / 7% / 10% |
| giant_popped/our1000/giant5000/d600/bot | 22+-4% / 28% / 6% | 22+-4% / 28% / 6% | 24+-5% / 32% / 6% | 28+-5% / 38% / 0% | 25+-5% / 30% / 0% | 24+-4% / 26% / 2% | 28+-5% / 38% / 4% |
| giant_popped/our500/giant3000/d600/bot | 38+-5% / 62% / 6% | 34+-5% / 52% / 4% | 36+-4% / 60% / 8% | 27+-4% / 30% / 6% | 27+-4% / 30% / 4% | 28+-4% / 30% / 6% | 33+-4% / 44% / 2% |
| giant_popped/our500/giant10000/d600/bot | 0+-0% / 0% / 4% | 0+-0% / 0% / 4% | 0+-0% / 0% / 4% | 0+-0% / 0% / 6% | 0+-0% / 0% / 6% | 0+-0% / 0% / 4% | 0+-0% / 0% / 4% |
| giant_popped/our500/giant5000/d1000/bot | 9+-4% / 6% / 0% | 5+-3% / 2% / 0% | 6+-4% / 6% / 0% | 15+-4% / 8% / 2% | 13+-4% / 10% / 4% | 3+-2% / 2% / 2% | 5+-3% / 2% / 2% |
| giant_popped/our500/giant5000/d1500/bot | 3+-2% / 0% / 0% | 2+-1% / 0% / 0% | 3+-2% / 0% / 0% | 15+-4% / 10% / 4% | 8+-4% / 6% / 2% | 2+-2% / 0% / 0% | 2+-2% / 0% / 0% |

(The 500/5000/600 row is N = 100, the others 50. Mass chain: 300/5000 26% and no overtake; 500/3000 100% and overtake; 1000/5000 100% and overtake; 500/10000 25% and no overtake.)

### A competitor (c3_4, 800 mass, at the same distance), N = 50

**giant_popped/our500/giant5000/d600/bot/comp800**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 0+-0% | 0+-1% | 1+-1% | 0.15+-0.01 | 0% [0%-7%] | - s | 10% / 10.1 s | 2% [0%-10%] | 0.5 |
| plan | 50 | 0+-0% | 0+-0% | 0+-0% | 0.15+-0.01 | 0% [0%-7%] | - s | 8% / 12.0 s | 0% [0%-7%] | 0.4 |
| harv_r25 | 50 | 0+-0% | 1+-1% | 1+-1% | 0.15+-0.01 | 0% [0%-7%] | - s | 14% / 12.6 s | 2% [0%-10%] | 0.5 |
| oracle | 50 | 0+-0% | 1+-0% | 1+-0% | 0.15+-0.02 | 0% [0%-7%] | - s | 32% / 14.9 s | 0% [0%-7%] | 1.1 |
| oracle_view | 50 | 0+-0% | 0+-0% | 1+-0% | 0.16+-0.02 | 0% [0%-7%] | - s | 32% / 15.4 s | 0% [0%-7%] | 1.0 |
| oracle_brave | 50 | 4+-1% | 5+-1% | 6+-2% | 0.02+-0.02 | 0% [0%-7%] | - s | 92% / 1.4 s | 84% [71%-92%] | 4.5 |
| p6 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.15+-0.01 | 0% [0%-7%] | - s | 12% / 10.2 s | 0% [0%-7%] | 0.4 |
| p3 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.15+-0.01 | 0% [0%-7%] | - s | 10% / 7.9 s | 0% [0%-7%] | 0.3 |

**giant_pop/our500/giant5000/d600/bot/comp800**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 1+-1% | 1+-1% | 2+-1% | 0.13+-0.02 | 0% [0%-7%] | - s | 16% / 13.7 s | 6% [2%-16%] | 1.1 |
| plan | 50 | 0+-0% | 1+-1% | 1+-1% | 0.13+-0.02 | 0% [0%-7%] | - s | 14% / 15.1 s | 2% [0%-10%] | 0.8 |
| harv_r25 | 50 | 1+-1% | 2+-1% | 3+-2% | 0.14+-0.02 | 0% [0%-7%] | - s | 30% / 9.6 s | 8% [3%-19%] | 1.9 |
| oracle | 50 | 1+-1% | 3+-2% | 7+-3% | 0.23+-0.07 | 6% [2%-16%] | 25.9 s | 60% / 8.4 s | 2% [0%-10%] | 4.2 |
| oracle_view | 50 | 1+-1% | 2+-1% | 5+-3% | 0.18+-0.05 | 2% [0%-10%] | 26.0 s | 66% / 9.9 s | 6% [2%-16%] | 4.0 |
| oracle_brave | 47 | 4+-1% | 5+-2% | 7+-2% | 0.06+-0.04 | 0% [0%-8%] | - s | 87% / 1.6 s | 83% [70%-91%] | 5.3 |
| p6 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.12+-0.01 | 0% [0%-7%] | - s | 8% / 21.4 s | 0% [0%-7%] | 0.4 |
| p3 | 50 | 0+-0% | 0+-1% | 1+-1% | 0.14+-0.03 | 0% [0%-7%] | - s | 10% / 13.6 s | 2% [0%-10%] | 0.9 |

**giant_popped/our500/giant5000/d600/bot/comp800**

| agent | edible piece in view (ticks) | ...of those: moving towards it | standing still | threat piece near (ticks) | ...of those: moving away | splits / rep | max cells | peak mass | lost to 3rd | stolen by comp |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 34% | 6% | 1% | 1% | 19% | 0.0 | 1.0 | 545 | 0 | 1207 |
| harv_r25 | 36% | 8% | 1% | 1% | 17% | 0.1 | 1.1 | 548 | 0 | 1220 |
| oracle | 47% | 19% | 3% | 2% | 27% | 0.0 | 1.0 | 547 | 0 | 1064 |
| oracle_view | 44% | 21% | 4% | 2% | 23% | 0.1 | 1.1 | 554 | 0 | 1101 |
| p6 | 26% | 3% | 1% | 1% | 16% | 0.0 | 1.0 | 536 | 0 | 1210 |
| p3 | 24% | 3% | 1% | 1% | 16% | 0.0 | 1.0 | 527 | 0 | 1195 |

**giant_pop/our500/giant5000/d600/bot/comp800**

| agent | edible piece in view (ticks) | ...of those: moving towards it | standing still | threat piece near (ticks) | ...of those: moving away | splits / rep | max cells | peak mass | lost to 3rd | stolen by comp |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 28% | 11% | 0% | 1% | 7% | 0.0 | 1.0 | 606 | 10 | 351 |
| harv_r25 | 38% | 19% | 3% | 1% | 9% | 0.0 | 1.0 | 643 | 20 | 361 |
| oracle | 81% | 34% | 2% | 6% | 36% | 0.0 | 1.0 | 866 | 0 | 247 |
| oracle_view | 77% | 30% | 2% | 8% | 46% | 0.1 | 1.1 | 767 | 0 | 249 |
| p6 | 17% | 4% | 0% | 0% | 2% | 0.0 | 1.0 | 534 | 0 | 457 |
| p3 | 22% | 6% | 0% | 1% | 7% | 0.0 | 1.0 | 612 | 10 | 433 |

### Other giants (`hunt` / `passive` scripted cursor, N = 50)

**giant_popped/our500/giant5000/d600/hunt**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 1+-1% | 2+-1% | 3+-2% | 0.14+-0.02 | 0% [0%-7%] | - s | 30% / 3.5 s | 6% [2%-16%] | 2.3 |
| harv_r25 | 50 | 1+-1% | 2+-1% | 3+-1% | 0.13+-0.02 | 0% [0%-7%] | - s | 34% / 6.5 s | 6% [2%-16%] | 2.4 |
| oracle | 50 | 1+-1% | 2+-1% | 3+-2% | 0.13+-0.02 | 0% [0%-7%] | - s | 32% / 4.1 s | 8% [3%-19%] | 2.5 |
| oracle_brave | 50 | 8+-2% | 8+-2% | 8+-2% | 0.00+-0.00 | 0% [0%-7%] | - s | 90% / 1.4 s | 100% [93%-100%] | 5.8 |
| p6 | 50 | 1+-1% | 2+-1% | 4+-2% | 0.16+-0.05 | 2% [0%-10%] | 28.9 s | 26% / 3.3 s | 6% [2%-16%] | 2.4 |
| p3 | 50 | 1+-1% | 2+-1% | 3+-3% | 0.16+-0.06 | 4% [1%-13%] | 29.3 s | 28% / 3.5 s | 4% [1%-13%] | 2.1 |

**giant_popped/our500/giant5000/d600/passive**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.11+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| harv_r25 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.10+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| oracle | 50 | 0+-0% | 0+-0% | 0+-0% | 0.10+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| oracle_brave | 50 | 8+-2% | 12+-3% | 12+-3% | 0.07+-0.07 | 4% [1%-13%] | 14.0 s | 82% / 3.7 s | 80% [67%-89%] | 6.8 |

**giant_pop/our500/giant5000/d600/hunt**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 1+-1% | 2+-1% | 3+-1% | 0.14+-0.02 | 0% [0%-7%] | - s | 26% / 7.2 s | 2% [0%-10%] | 2.1 |
| harv_r25 | 50 | 1+-1% | 2+-1% | 3+-1% | 0.13+-0.02 | 0% [0%-7%] | - s | 28% / 7.5 s | 2% [0%-10%] | 2.2 |
| oracle | 50 | 1+-1% | 3+-2% | 5+-3% | 0.18+-0.06 | 4% [1%-13%] | 29.1 s | 36% / 7.6 s | 4% [1%-13%] | 3.5 |
| oracle_brave | 49 | 8+-3% | 8+-3% | 8+-3% | 0.03+-0.05 | 2% [0%-11%] | 7.8 s | 88% / 2.3 s | 94% [83%-98%] | 5.5 |
| p6 | 50 | 0+-0% | 1+-1% | 1+-1% | 0.12+-0.01 | 0% [0%-7%] | - s | 14% / 8.1 s | 0% [0%-7%] | 1.1 |
| p3 | 50 | 0+-0% | 1+-1% | 2+-2% | 0.15+-0.05 | 2% [0%-10%] | 26.6 s | 14% / 7.4 s | 0% [0%-7%] | 1.4 |

**giant_pop/our500/giant5000/d600/passive**

| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.11+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| harv_r25 | 50 | 0+-0% | 0+-0% | 0+-0% | 0.11+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| oracle | 50 | 0+-0% | 0+-0% | 0+-0% | 0.10+-0.00 | 0% [0%-7%] | - s | 0% / - s | 0% [0%-7%] | 0.0 |
| oracle_brave | 49 | 8+-2% | 12+-3% | 12+-3% | 0.03+-0.05 | 2% [0%-11%] | 12.9 s | 82% / 4.1 s | 88% [76%-94%] | 6.6 |

`passive` is a null control: the pieces regroup around the core within a second and every agent, the oracle included, eats nothing
(only the brave oracle gets its 5-6% in the first seconds and dies). Pieces are only reachable in the first ~1-2 s after the burst and while a hunting giant drags them to you.

### Behaviour counters (diagnosis, headline cells)

**giant_popped/our500/giant5000/d600/bot**

| agent | edible piece in view (ticks) | ...of those: moving towards it | standing still | threat piece near (ticks) | ...of those: moving away | splits / rep | max cells | peak mass | lost to 3rd | stolen by comp |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 69% | 8% | 1% | 14% | 48% | 0.0 | 1.0 | 1136 | 0 | 0 |
| plan | 68% | 8% | 1% | 13% | 46% | 0.1 | 1.1 | 1080 | 0 | 0 |
| harv_r25 | 73% | 10% | 1% | 13% | 48% | 0.1 | 1.1 | 1197 | 0 | 0 |
| plan_harv_r25 | 68% | 10% | 1% | 11% | 53% | 0.2 | 1.2 | 992 | 0 | 0 |
| oracle | 81% | 10% | 1% | 18% | 64% | 0.1 | 1.1 | 1343 | 0 | 0 |
| oracle_view | 80% | 10% | 1% | 17% | 63% | 0.1 | 1.1 | 1345 | 0 | 0 |
| oracle_brave | 93% | 67% | 2% | 66% | 23% | 0.5 | 1.5 | 879 | 0 | 0 |
| p6 | 69% | 4% | 1% | 13% | 44% | 0.0 | 1.0 | 1062 | 0 | 0 |
| p3 | 73% | 9% | 1% | 19% | 57% | 0.0 | 1.0 | 1223 | 0 | 0 |

**giant_pop/our500/giant5000/d600/bot**

| agent | edible piece in view (ticks) | ...of those: moving towards it | standing still | threat piece near (ticks) | ...of those: moving away | splits / rep | max cells | peak mass | lost to 3rd | stolen by comp |
|---|---|---|---|---|---|---|---|---|---|---|
| c3_4 | 30% | 11% | 0% | 3% | 14% | 0.0 | 1.0 | 765 | 0 | 0 |
| plan | 28% | 11% | 0% | 3% | 14% | 0.1 | 1.1 | 722 | 0 | 0 |
| harv_r25 | 37% | 20% | 4% | 4% | 19% | 0.1 | 1.1 | 749 | 0 | 0 |
| plan_harv_r25 | 36% | 21% | 4% | 3% | 18% | 0.2 | 1.2 | 696 | 0 | 0 |
| oracle | 76% | 12% | 1% | 14% | 58% | 0.0 | 1.0 | 1231 | 0 | 0 |
| oracle_view | 74% | 13% | 1% | 13% | 57% | 0.1 | 1.1 | 1197 | 0 | 0 |
| oracle_brave | 93% | 69% | 3% | 62% | 20% | 0.3 | 1.3 | 861 | 0 | 0 |
| p6 | 21% | 9% | 0% | 1% | 4% | 0.0 | 1.0 | 599 | 0 | 0 |
| p3 | 27% | 11% | 0% | 3% | 10% | 0.0 | 1.0 | 740 | 0 | 0 |

## Diagnosis

Why does the best agent fall short of the oracle, and of the mass-chain bound?

1. **The oracle is a baseline, not an upper bound.** It is safety-first and loses to c3_4 where fear costs more than it saves (500 vs 3k: oracle 27% capture / 30% overtake, c3_4 38% / 62%;
   it is also no safer: 12% deaths in 30 s against c3_4's 7%). The only hard bound is the mass chain (51% / overtake for 500 vs 5k). Closing the gap to that is a different problem: out-running a hunting giant for 30 s.
2. **`giant_popped`: no significant gap.** c3_4 13+-2, p3 14+-2, p6 11+-2 vs oracle 17+-3 at 30 s (N = 100); to resolve 3pp you need N ~ 400. Every agent takes the
   easy ~13% (12 small pieces + the 318) in the first 10-20 s and then stalls: the 637 piece (needs 842 mass, we have ~1150) and the 1275 one (needs 1686) move about as fast as the hunter (6.7 and 7.8 units/tick for the 1275 and 637 pieces, against 8.2 for the 500 hunter and 6.8 once it has grown to 1150), are
   steered by a giant that is hunting the hunter, and a lunge by a split core is deadly. All agents spend 13-19% of the ticks with a threatening piece within 400 and move away in 44-64% of those: the cost of the giant, not of the agent.
3. **Too slow to commit: not in `giant_popped`.** The first piece comes after 1.5-1.9 s for every agent (95% of the reps). p6 is a little slower at the start (5% at 10 s vs 8%).
4. **Flees / is not there when it pops: this is the big one (`giant_pop`).** The rule bots and nets retreat from the approaching 5k giant (centre distance at the pop 1914 for the rule bots, 1931 p6, 1905 p3, against 1566 for the oracle, 1406 for the brave one; with no movement it would be 1273), then see an edible piece in only
   21-37% of the ticks (oracle 75%), take 8-14 s to the first piece and only 13-32% ever get one in 30 s (oracle 93%, 2.8 s). Capture 1-5% vs 15%. p6 is the most timid (1%, first piece in 13% of the reps),
   p3 in between (4%). The view box (about 3200 x 1800 for a 500 cell) and the lack of memory matter at larger D: at D = 1000/1500 c3_4 falls to 9%/3%, the nets to 3-5%/2%, the full-view oracle holds 15%/15% while `oracle_view` falls to 13%/8%.
   `harvest_range` 2.5 helps the right thing (edible piece in view 30 -> 37%, moving towards it 11 -> 20% of the time) but not the outcome (5%).
5. **Does not split onto pieces: not the issue here.** Nobody splits (0.0-0.2 per rep: any piece within a lunge of a 28-mass bit is also within reach of the core) and `oracle_nosplit` is identical to `oracle` (17 vs 17%, 15 vs 15%).
   The brave oracle (no fear, splits at will: 0.3-0.5 splits/rep) is eaten in 87-91% of the reps and captures 7-8%.
6. **The harvest genes do not move this metric.** `harvest = 1`: 13+-3 vs 13+-2; `+ harvest_range` 2.5/4: 14+-3 (2.5 and 4 are identical, the pieces beyond gap x2.5 are out of view anyway). The planner does not help (11+-2),
   planner + harvest hurts a little (9-10+-2).
7. **Competes badly: everybody does.** With an 800-mass c3_4 at the same distance the competitor takes 1060-1220 of the giant's ~5100 mass (21-24%) and we get 0-1% in `giant_popped`
   (even the oracle: 1%); in `giant_pop` the oracle keeps 5-7%, rule bots 2-3%, nets 0-1%. Deaths to the competitor are rare (the agents keep away from an 800 cell), the loss is the pieces.
8. **The hunter must be big enough.** 300 vs 5k: 0-1% for everyone (the bound is 26%); 500 vs 10k: 0%. 1000 vs 5k: 22-28% capture and 26-38% overtake; 500 vs 3k: 27-38% and 30-62%. The user's "500 beats 5k"
   holds at the 5-12% level within 30 s (28-44% within 60 s) with the best agents, and costs a death in 7-12% (30 s) to 12-34% (60 s) of the reps.

What would raise the numbers (not tried): approach pieces from the pop point as a group (not the nearest), keep a memory of pieces that left the view, chase pieces away from the core instead of fleeing it, use the 637/1275 pieces' slower speed when they are split (a lunge needs the half cell to still eat), and learn this from the drill (it is a cheap, seeded, shaped reward).

## Not done / caveats

* Only a c3_4 giant (and a c3_4 competitor) as the opponent; the giant never plays a human's game (no deliberate regrouping, no virus tricks beyond c3_4's). `hunt`/`passive` are scripted and degenerate.
* **`giant_split_pop`** (the giant splits to attack and leaves halves) is not implemented: its halves (1250+) are not eatable by a 500 hunter, so it needs a different hunter mass and a decision on how the giant splits.
* **Vision-reflex variants** of vision-m5 are not on master, not run. The nets ran on CUDA; one batch of a 6-thread run crashed in cuDNN and was rerun in chunks of 100 reps with 4 threads (each chunk is a separate file, rerun on failure).
* Sweeps use N = 50 and only the `giant_popped` scenario; one seed base (1). The oracle's constants (200/300/100) were tuned for a few minutes on the 500/5k cell, not searched.
* The scene differs from `ogar endless`: no server bots, no stray viruses, a map of 7000. Mass eaten from the giant by pieces merging into other food is not counted, and decay (0.2%/s) hits everyone.
* `capture` counts mass of eaten cells at the moment of eating; our cells' own mass loss to the giant is `lost_giant` in the JSON rows.

## Lurk-and-harvest hunter (results, 2026-10-09)

Code: `rust/arena_client/src/hunter.rs` (module + 21 unit tests on constructed snapshots), genes in `params.rs`, hooked into `HeuristicPolicy::act`
(policy.rs; the planner defers to it while it acts), scripted drill agent `hunter[:gene=v,...]`. All genes default to 0 (off).
Specs: `bot/drill/hv_commit_only.json` (the recommended set: c3_4 + `commit commit_kite split_harvest race hv_memory=3 hv_still=100`),
`bot/drill/hv_full.json` (+ `lurk`), `plan_*` wrappers. `OGAR_HUNTER_TRACE=1` prints the decisions.

Behaviours: **lurk** (giant >= `lurk_ratio` x our mass within `lurk_virus_range` of a virus: stand at lunge reach + `lurk_margin` from it, on the far side of the
virus; abort to the normal bot when it closes on us or another threat appears; `lurk_shoot` feeds the virus into it via the existing shoot tactic with
`shoot_threats`); **commit** (burst = a group of >= `hv_min_pieces` same-colour cells with `hv_group_ratio` x our mass and something edible; `commit_kite`: 32 headings
played out for `hv_horizon` steps against pursuing eaters, thrown pieces on their boost path and lunge circles, take the safe one that eats most mass soonest;
`commit_order` 1 = smallest first); **split-harvest** (split only if our half eats the piece, the throw lands it, and no cell that could eat either half
reaches it or the walk; `hv_split_frac` min piece size; the merge timer is 30 s = longer than any drill, so it is judged safe for its whole life);
**race** (pieces a rival that eats them reaches first are worth nothing); **memory** (burst pieces that left the view are kept `hv_memory` x 10 steps).
Real split reach: `780 * (size/sqrt2)^0.0122` (thrown half) + the half's radius; a threat can lunge only if its half still eats the victim.

### Drill (500 vs 5k, bot giant, seed base 5000, N = 100, 95% intervals; genes chosen on other seeds)

30 s, capture at 30 s / overtake any time / died:

| cell | c3_4 | plan | oracle | scripted hunter | hv_commit_only | hv_full (lurk) |
|---|---|---|---|---|---|---|
| giant_popped D600 | 12+-2 / 6% / 9% | 11+-2 / 6% / 9% | 17+-3 / 15% / 9% | 23+-3 / 23% / 9% | **24+-3 / 27% / 10%** | 24+-3 / 27% / 10% |
| giant_popped D1000 | 5+-2 / 1% / 2% | 4+-2 / 1% / 0% | 13+-3 / 10% / 7% | 23+-3 / 23% / 2% | 19+-3 / 17% / 3% | 19+-3 / 17% / 3% |
| giant_popped D1500 | 3+-1 / 0% / 1% | 2+-1 / 0% / 1% | 14+-3 / 11% / 6% | 22+-3 / 21% / 4% | 13+-3 / 12% / 1% | 13+-3 / 12% / 1% |
| giant_pop D600 | 5+-2 / 3% / 3% | 3+-1 / 1% / 2% | 18+-3 / 16% / 11% | 20+-3 / 15% / 6% | 15+-3 / 13% / 7% | **20+-3 / 19% / 7%** |
| giant_pop D1000 | 4+-2 / 1% / 1% | 2+-1 / 0% / 1% | 14+-3 / 9% / 5% | 21+-3 / 18% / 2% | 11+-3 / 10% / 2% | 15+-3 / 13% / 6% |
| giant_pop D1500 | 3+-1 / 0% / 1% | 2+-1 / 0% / 0% | 13+-3 / 7% / 8% | 18+-3 / 14% / 2% | 9+-3 / 9% / 1% | 12+-3 / 11% / 4% |

60 s, D600 (capture at 60 s / overtake / died): giant_popped c3_4 17+-3 / 18% / 19%, oracle 26+-4 / 38% / 28%, **hv_commit_only 33+-4 / 52% / 17%**, scripted hunter 32+-4 / 47% / 22%;
giant_pop c3_4 8+-3 / 11% / 10%, oracle 29+-4 / 43% / 24%, hv_full 27+-4 / 36% / 17%, hv_commit_only 23+-4 / 30% / 17%.
Other masses (giant_popped, 30 s, capture / overtake): ours 300: c3_4 0 / 0, hunter genes 4+-1 / 0 (nobody; the chain bound is 26%); ours 1000: c3_4 27 / 38%, hv_commit 37+-3 / 66%;
ours 1500: 31 / 51% -> 37 / 68%; giant 3k: 38 / 66% -> 42 / 76%; giant 10k: 0 -> 2 / 0.
With an 800 competitor (D600): giant_pop c3_4 2+-1 -> hv_full 11+-2 (overtake 0 -> 5%), oracle 7; giant_popped everyone 0-2% (the competitor takes the pieces; `race` does not fix it).

Targets: **>= 50% overtake in 60 s: met in giant_popped (52% [42-62]), not in giant_pop (36%). >= 35% capture: not met** (33+-4% at 60 s, 24+-3% at 30 s). Deaths did not rise
(popped 17% vs 19% at 60 s; giant_pop 17% vs 10%, higher). Why the rest is not reached: the core (2521) holds a lunge circle of about 1190 around us that we respect,
so 637/1275 pieces are only caught when they drift to us; a random search over the genes (30 candidates, 60-s capture) topped out at about 31% (baseline 27% on the same seeds), so the gene values are near a plateau of this design, not tuned to the limit.
Deaths at t = 1 s (a thrown 1275/2521 piece landing on us) are the biggest single cause of deaths; `landing_of` dodges it only after the first step's velocity is seen.
Rule bots only see their view box: at D >= 1000 in `giant_popped` the pieces are never seen, which is why the scripted hunter (oracle fallback walks to unseen pieces) beats the rule-bot variants there.

### Endless leader share (`arena.endless --engine inproc`, usual rule-bot field: champ, r2g9, safe, ramp_r2, ramp300 x2 each + 8 server bots; candidate x2; 8 worlds x 240 min)

| candidate | 3000 | 11000 | 21000 | 31000 | 41000 | mean | deaths/h | harvest_mass/h |
|---|---|---|---|---|---|---|---|---|
| c3_4 | 31 | 28 | 31 | 32 | 36 | 31.6 | 16.0 | 14.6k |
| c3_4 + hv_commit_only | **43** | **44** | **38** | **37** | **37** | **39.8** | 14.6 | 18.6k |
| c3_4 + hv_full (lurk) | 42 | 40 | 32 | 32 | 31 | 35.4 | 15.7 | 19.8k |

Counters from `--stats-out` (candidate rows, 320 player-hours each; the sum of the 5 sets; `rl/vision/skillstats.py` keys). Gate (mean improves, >= 3 of 5): **hv_commit_only passes (5/5, +8.2)**;
hv_full wins 3 of 5 strictly (one tie at 32, one loss) with a smaller mean gain: **lurk does not pay on endless, keep it off** (it helps `giant_pop` in the drill only).
The coordinator's M5 warning (a harvest reflex that chases pieces into the crowd that popped the target raised deaths 57%) does not show up here: deaths/h fell 9% (16.0 -> 14.6) because
every step re-plays the pursuers (all cells that outweigh us, whatever their owner), thrown pieces and lunge circles, and `race` drops pieces a rival reaches first. "Mass lost per hour" is not in the counters, so it was not compared.
Not run: the held-out field (it needs vision nets; for rule bots the usual field applies), `plan_c34` variants in endless (the planner defers to the hunter, drill identical), `lurk_shoot` end to end (only a unit test), more than one seed base per set.

### Hook for the vision agents (do not edit visionbot.rs here; vision-m5 owns it)

    use arena_client::hunter::{Hunter, Phase};
    let genes = Params::from_json(&serde_json::json!({"commit":1,"commit_kite":1,"split_harvest":1,"race":1,"hv_memory":3,"hv_still":100}));
    let mut hunter = Hunter::new();                       // one per agent; hunter.reset() on respawn
    // every decision step, with the Snapshot the rule-bot teacher already gets (ogar::agent::snapshot(&world, pi)):
    if let Some(out) = hunter.step(&snap, &genes) {       // Out { x, y, split, eject }: the aim point and the split bit
        // teacher: label this step with out; reflex: override the net's aim/split while hunter.phase == Phase::Commit
    }

`step` is pure apart from velocities/cooldowns/memory kept in `Hunter`; it returns `None` when there is no burst giant (or lurk target), so the net plays on.
Use the `commit` set only (`lurk` lost on endless). A reflex must keep the hunter's own safety (it already refuses to approach inside any eater's reach or lunge circle); do not add chasing of
pieces that it did not pick.
