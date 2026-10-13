> Historical: describes code removed from the repo; see [README.md](README.md).

# Training Plan — local, faster-than-real-time, evolution + RL, Jev on top

Goal: train agar.io bots fully locally by running many bots against each other faster than real time. Improve them by tournament-driven evolution, then by RL self-play. Keep Jev (or local Kev) as an optional strategic layer, and measure whether it helps.

## Findings that shape this

- The browser bot is limited to real time: one window per bot, a 60 ms mouse cadence and a 40 ms server tick. Fine for watching, useless for training.
- MultiOgar's physics is a **fixed 40 ms step**: `GameServer.timerLoop` (src/GameServer.js:505) sleeps until the next step, then calls `mainLoop`. Patching out the sleep gives **faster-than-real-time** play with identical physics, limited only by CPU.
- Hosted Jev takes 220–410 ms and is billed per call. It **cannot sit inside a training loop** (millions of steps). During training the strategic layer must be either absent, simulated, or a local model (Kev-4B on the RTX 5090, ~20 ms).
- First A/B (2026-10-03, about 4 min, same server): pellet route planner reached mass 159 with 0 deaths; the old potential field reached 114 with 2 deaths. One sample only, so it's not significant yet. The tournament harness below exists to make comparisons like this rigorous.

## Phase 1: Headless arena (foundation): DONE 2026-10-03

What exists:
- `arena/server/arena.js` launches the vendored server and patches it at runtime (vendor files untouched). It adds CLI config, a seeded `Math.random`, and lockstep stepping: agents send 0xC8 to register, the server sends 0xC9 <tick> after every `--sync` ticks, and agents ack with the same 0xC9 packet once they've acted. It also turns off the IP limit, coordinate scrambling, wall-clock timeouts and the "minion" check, which flags non-browser clients.
- `arena/client.py` is a headless protocol-6 client. It keeps a cell table from UpdateNodes deltas and tracks per-agent stats: avg mass (mass-time integral / ticks), peak mass, deaths, kills.
- `arena/policies.py` holds the policies (`route`, `field`). They reuse `bot/steer.py` + `bot/farm.py`, with `bot/modes.py` standing in for Jev (rule-based flee/hunt/farm).
- `arena/match.py` runs one match (`--watch` opens a spectator window, `--max-tps 50` = 2x realtime). `arena/tournament.py` runs N seeded matches in parallel, one server per match on a free port, and reports a 95% CI over matches.

Measured:
- **Deterministic**: the same seed gives identical results, so variants can be compared on identical matches.
- Speed: one match with 8 agents runs at 7.6x realtime (the Python policies are the bottleneck). 6 workers give **28x realtime aggregate** (48 game-min in 103 s).
- First tournament, route:4 vs field:4, 24 matches × 2 game-min: avg mass **50.4 ± 6.2 vs 50.3 ± 4.3**, which is **no significant difference**. The earlier single-run "route wins" was noise. Field peaks and kills slightly higher, within the CIs.

Original plan for this phase:

- `arena/` Python client that speaks the MultiOgar WebSocket protocol directly, with no browser. Packets are the ones the client uses: 0x00 play, 0x10 mouse, 0x11 split, plus parsing of 0x10 node updates and 0x11/0x12/0x14.
- Server patch `--fast`: no sleep in `timerLoop`, configurable map size, food and bot counts, plus a seed for reproducibility.
- Run N servers (one per CPU core; the 5800X has 8 cores, so about 6–7 arenas) × M bots each.
- Spectate: one browser window on any arena with the client's spectate mode, so you can watch matches live.
- Metrics per bot per match: peak mass, mass-time integral, survival time, kills, deaths.

## Phase 2: Tournament + evolution of the hand-written bot (quick win)

- Expose every magic number as a genome: steering weights, danger radius margin, `SPLIT_RANGE`, `EAT_RATIO` safety margin, route `CANDIDATES`/`DEPTH`/`SWITCH_MARGIN`, split policy thresholds. About 15–25 floats.
- Population of about 32 genomes. Each generation is a round-robin or Swiss tournament across arenas, with fitness = mean mass-time integral, plus TrueSkill/Elo for head-to-head ranking.
- Optimizer: CMA-ES, or a simple GA (tournament selection, Gaussian mutation, elitism). Keep a hall of fame of past champions in every arena so bots don't just overfit to the current population.
- Should beat the hand-tuned bot within hours of compute.

## Phase 3: Neural policy + RL self-play

- Observation: egocentric grid around the own cell (e.g. 64×64). Channels: food, own cells, other players as signed log mass ratio, viruses, ejected mass, borders. Plus scalars: own mass, cell count, merge timers.
- Action: direction (16–32 discrete headings or continuous), split yes/no, eject yes/no. Decided every 2 server ticks.
- Algorithm: PPO (CleanRL/PufferLib style), with the CNN policy on the 5090.
- Self-play **league**: the current policy, frozen snapshots, evolved Phase-2 bots and MultiOgar's built-in bots as opponents. Population-based training mutates hyperparameters and copies winners, which is the "tournament + evolution" idea applied to RL.
- Throughput option: if the Node server becomes the bottleneck, port the core rules (movement speed vs size, eat ratio, split/merge, viruses) into a vectorized NumPy/JAX simulator for bulk training. Then validate and fine-tune on real MultiOgar. Decide based on measured steps/s from Phase 1.
- Bootstrap: behaviour cloning from the Phase-2 champion first, so PPO doesn't start from random wandering.

## Phase 4: Jev / Kev on top

- The policy takes a **mode input** (farm / hunt target / flee / hide) as extra observation channels. During training the mode is set by a cheap heuristic or randomized, so the policy learns to follow it.
- At play time the strategic layer picks the mode: Jev hosted, Kev-4B local, or the heuristic. Same `/v1/systemone` interface, switched by base URL.
- Also use it for rare, high-stakes calls (split-attack go/no-go, when to hide), at most a few per second.
- **Measure it**: tournament runs with the strategic layer on vs off, same policy. Keep the layer only if it wins. Expectation: once a trained policy handles reflexes, an LLM layer adds little. That's still worth testing, and it's cheap with local Kev.

## Order and rough effort

| Phase | Effort | Output |
|---|---|---|
| 1 Headless arena + fast server + spectate | 1–2 days | N parallel matches, metrics, watchable |
| 2 Evolution of hand-written bot | 1 day + compute | Tuned champion, ranked leaderboard |
| 3 RL league (PPO + self-play) | 1–2 weeks | Neural bot that beats the champion |
| 4 Strategic layer A/B | 1–2 days | Evidence on whether Jev/Kev helps |

## Status 2026-10-03 (late): saturation and RL

**Evolution:** r1 → r2 → r3 (league) → r4 (league + virus-skill genes). r4 ran 8 generations
with no significant improvement over its start (r3 champion_008), so the parameter space is
**saturated**. Virus shooting/hiding genes stayed off (evolution found no benefit).

**Fresh-seed tournament (24 matches, 4 game-min, seeds 31000+):** every champion from r1g7 onward
sits within ~0.3 log-mass (4.93–5.26), with overlapping CIs. Best: **r2g9 (5.26)**, which beats the
"latest" r3g8 (5.06) in 67% of matches. Baselines: route 3.86, field 4.24.
Lesson: promotions decided on the *same* 18 validation seeds every generation suffer from the
winner's curse; later "improvements" were partly seed luck. Fix for future runs: fresh
validation seeds per duel, plus a final fresh-seed tournament before declaring a champion.

**RL:** BC from the champion (67% move accuracy) ≈ champion level. Unanchored PPO (p1–p3)
degraded play: noisy per-step reward, policy entropy drift. What worked (p4): telescoped
reward (Δ log-mass, death = −log-mass), separate critic, critic warm-up, 24.5k-sample batches,
lr 5e-5, and a KL anchor to the BC policy. p4 reaches champion level (best checkpoint
it36: +0.01 vs champion on 6 matches; 4.86 in the 24-match tournament) but has not surpassed
the evolved champions. Split/W are explored only above mass 200 (cheap exploration floor).

## Final result (2026-10-03, 23:15)

40-match fresh-seed confirmation (seeds 41000+, 4 game-min):

| bot | log mass | beats r1g7 | beats nn36 | deaths/match |
|---|---|---|---|---|
| champ = r3 champion_008 (= r4 final) | 5.04 ± 0.09 | 82% | 75% | 0.92 |
| r2g9 | 5.06 ± 0.10 | 78% | 75% | 0.93 |
| r1g7 | 4.66 ± 0.12 | – | 52% | 1.80 |
| nn36 (PPO p4 it36) | 4.63 ± 0.15 | 48% | – | 2.59 |

champ and r2g9 are statistically tied (45/55) and clearly beat all other bots (and the
baselines ~90%). No method improved on them: r4 (8 generations incl. virus skills) and
PPO p4 (~120 iterations past its best checkpoint) both plateaued. **Final champion:
runs/final_champion.json** (r3 champion_008). PPO state kept at runs/rl/p4_final.pt
(resume: `python -m rl.ppo --run p5 --init runs/rl/p4_final.pt --anchor runs/rl/bc.pt ...`).

## 2026-10-04: QD search and guided NN g6-g8 (plateau)

- **QD (arena/qd.py, run q1):** MAP-Elites over playstyles. It found 14/27 niches, among them virus shooters, self-feeders and double-splitters. None of 8 champion duels (12 matches each, fresh seeds) beat runs/final_champion.json. It saturated at round 29.
- **Guided NN:**
  - g6: lr 3e-4. It peaked around iteration 54, then drifted down.
  - g7: lr 1e-4, anchored to g6 ckpt54.
  - g8: from g7 ckpt57, with `--strong --opp-every 1` (only the champion and its hunter as opponents, deciding every step). Its training score was flat from iteration 50.
  - Large fresh-seed 3-way tests: g5 ckpt90, g6 ckpt54, g7 ckpt30/57 and g8 ckpt48/99 all tie the champion (44-58% win). Single 24-36-match "wins" did not replicate.
- **Final field test (36 matches; runs/rl/final_nn.pt = g8 ckpt48):**

  | Bot | log-mass |
  |---|---|
  | nn | 5.44 ± 0.15 |
  | champ | 5.51 ± 0.16 |
  | r2g9 | 5.58 ± 0.26 |
  | r1g7 | 5.13 |
  | nn36 | 4.84 |
  | route | 4.25 |
  | field | 4.51 |

  The NN ties the top rule-based bots (44-50%) and beats everything else (64-89%).
- **Why it caps:** the guided net executes the champion's suggestion about 97% of the time. PPO only learned small deviations: fewer deaths, no extra mass. Beating the champion needs a policy that acts on its own more, for example a lower follow prior or penalty, a population/self-play league, or much more experience (a vectorised simulator).

## 2026-10-04 (later): real-server fixes, then the switch to the simulator

- **Virus steering (bot/steer.py, gene `virus_detour`):**
  - The champion used to freeze when food pulled it toward a virus it would pop on. It was stuck 25% of the time, 24% of it at a virus (`arena/stuck.py`).
  - It now steers around the nearest blocking virus, on the near side. That cut stuck time to 2.6%.
  - In 36 matches the new champion beat the old one 69% and nnbest 69% (avg mass 306 vs 231 vs 264). It is the default now.
- **Big-cell hunting (gene `big_hunt`):** grow hunt range and freeze the prey floor above 1500 mass. It **lost** to plain farming 33-67 in 24 long 2x-food matches, so it is off by default.
- **g9-g11** (follow cost, tactics guide, virus reward, league):
  - The follow rate stayed at 97-98% and the net stopped changing.
  - g10/g11 were stopped. Their last checkpoints are runs/rl/g10/, runs/rl/g11/.
- **Switch to the GPU simulator:** pure self-play run s1, then evolution e1. See [sim-training.md](sim-training.md).
- **The real server is shut down for now.** To bring the showcase back, see [playing.md](../playing.md) (the Node showcase described here is gone).

## 2026-10-06 (night): rule bots in long games, split-safety genes

- **Long games** (15 game-min, 12 fresh-seed matches; `runs/bench/long_72000.json`): champ, nnbest and r2g9 tie (log-mass 6.62-6.72, all 42-58% head to head), each dying ~4 times per 15 min, mostly to each other.
  - About half of the big losses (chunks over 300 mass, while over 1000 total) happen while the loser is split.
  - A quarter of champ's go to players smaller than it overall, i.e. they eat its pieces.
  - Champ judges threats only relative to its largest cell.
- **New genes** (off by default):
  - `half_split_safety`: judge split safety against the post-split halves.
  - `piece_threats`: push away from every piece's threats, by mass share.
- **Result:** `half_split_safety` won 24 matches (+0.18, half the deaths) but **did not replicate** on 36 fresh seeds (5.38 vs champ 5.42). `piece_threats`: no effect.
- **The 4-min format can't separate the top bots.** champ, r2g9, nnbest and hs all score 5.38-5.47 +-0.13.
- **Showcase long run** (8.5 game-hours since the restart): nnbest 3223, champ 3131, r2g9 2174, g4net 1657, evader 1055, ffa (guarded net) 964. Humans average ~9-11k, mostly by dying 2-3x less often.

### The cautious champion beats every other bot in long games

`runs/champ_safe.json` (copy: `bot/champ_safe.json`) is the final champion with three changes: `flee_margin` 0.2 -> 1.0 (flee at the full danger radius), `split_safety` 0.5 -> 2.0, and `half_split_safety` on. In 15-game-min real-server matches (2x food, 8 MultiOgar bots), two independent fresh-seed runs of 14 matches:

| run | safe | champ | r2g9 | nnbest | g4net | safe's deaths vs others' |
|---|---|---|---|---|---|---|
| seeds 75000 | **6.85** | 6.18 | 6.40 | 6.40 | - | 2.75 vs 4.96-5.96 |
| seeds 76000 | **6.90** | 6.26 | 6.52 | 6.56 | 6.30 | 2.39 vs 4.43-5.64 |

safe's head-to-head wins over both runs: champ 75%, r2g9 64%, nnbest 75%, g4net 79%. Its average mass is ~1400, against at most ~1070 for the others.

In long games, dying is what costs: each death throws away accumulated mass, and the rule bots were tuned in 4-min matches where recklessness pays. A half-way variant (`flee_margin` 0.6, `split_safety` 1.2) did not help (6.22).

**Update: not consistent.** A third run in the same field (seeds 79000) tied: safe 6.54, champ 6.71, g4net 6.73.
- Pooled over 42 matches, safe still leads (~6.76 vs 6.38-6.52; head to head 67% champ, 62% r2g9, 69% nnbest, 57% g4net), but a single 14-match run can go either way.
- In a field of three cautious variants, the aggressive r2g9 won (71% over safe): caution pays most against reckless opponents.
- `r2g9_safe` (r2g9's genome with the same caution changes, 28 matches): 6.48, a tie (field 6.40-6.71).
- Long games are very noisy (+-0.3 per 14 matches), and all champion-family bots tie within it. A consistent winner needs a qualitatively different player, not gene tweaks.

**Pooled over 70 long matches** (4 standard-field runs; `uv run python -m arena.pooled runs/bench/safe_long_*.json`):

| bot | log-mass |
|---|---|
| safe | 6.73 +-0.14 |
| g4net | 6.51 |
| nnbest | 6.50 |
| r2g9 | 6.47 |
| champ | 6.39 |

| safe vs | wins | 95% Wilson interval |
|---|---|---|
| champ | 64% | 53-74% |
| r2g9 | 64% | 53-74% |
| nnbest | 63% | 51-73% |
| g4net | 55% | 42-68% (n=56, not significant) |

safe consistently beats champ, r2g9 and nnbest in long games, but not yet g4net. In 4-min games it ties the field, with half the deaths.

Its remaining leaks: ~1.5 own virus pops per 15 min, and 36% of its big losses are whole-on-whole (caught by a bigger cell).

**Searching around safe** (`arena.long_search`: CMA-ES over 15 genes; long games vs champ x3 + r2g9 x3; 10 generations of 6 candidates x 6 matches):
- Per-candidate scores are too noisy for CMA to make headway.
- Re-checking the top 6, the mean genome and safe on 14 fresh seeds, nobody beat safe clearly: best cand +0.52, safe +0.39, mean genome +0.10 (margin over the field).
- `long1` (the best, `runs/champ_long1.json`) in the full field (28 matches): 6.65, the top score with the fewest deaths, but only 46% vs r2g9 in that run.

### Duels (one on one, long games)

The full field mixes interactions: cautious bots profit from reckless ones feeding each other. So each candidate x2 now plays one opponent x2, plus the 8 MultiOgar bots, for 28 fresh-seed 15-min matches.

**safe** (`bot/champ_safe.json`):

| vs | wins | 95% interval |
|---|---|---|
| champ | 75% | 55-88% |
| nnbest | 79% | 60-90% |
| g4net | 86% | 65-95% |
| r2g9 | 44% | 27-63% |

r2g9 kills 67 per match to safe's 37: its hunting out-grows safe's survival.

**Caution on r2g9's genome**, vs r2g9:

| variant | wins | 95% interval |
|---|---|---|
| `safe_r2` (always cautious) | 54% | 36-70% |
| `ramp_r2` (`bot/r2g9_ramp800.json`: r2g9's genes, cautious above 800 mass) | **75%** | 57-87% |

ramp_r2 is now dueling champ, nnbest, g4net and safe.

**ramp_r2 duels** (seeds 88000):

| vs | wins | 95% interval |
|---|---|---|
| champ | 79% | 60-90% |
| r2g9 | 75% | 57-87% |
| g4net | 68% | 49-82% |
| nnbest | 54% | 36-70% |
| safe | 43% | 27-61% |

Neither new bot beats all four existing bots: safe fails against r2g9, ramp_r2 against nnbest. Next: r2g9's genes, cautious from 300 mass (`runs/r2g9_ramp300.json`).


### c3_4: the first bot that consistently beats all the others (2026-10-07)

`bot/c3_4.json` = ramp300 (r2g9's genes + caution ramp) + the wall genes (`wall_danger` ~370, `wall_wary` ~430: react earlier when pinned on a wall; keep off walls while a bigger player is near) + a CMA-ES search from there on the Rust engine (`arena.long_search --engine ogar`), which moved caution to ~790 mass, a smaller danger margin and more threat avoidance while farming.

How it got there:
- Loss mining (`arena.losses`): 82% of the rule bots' lost mass was lost at walls -> wall genes (ww400 beat 6 of 7 bots on Node, not ramp_r2).
- Fast evaluation: the Rust bot client (`--rust`) and the Rust MultiOgar port (`--engine ogar`, ~500 long matches/h/core) made gene search and A/Bs affordable; the Rust engine overrates candidates vs ramp_r2, so finalists are judged on Node.
- Bug fixes found on the way: corner shaking while hunting, split pieces jamming (mouse 600 ahead of the main cell), wall-clock exploration memory, and split pieces popping on viruses (`piece_detour`, no gain on Node).

**Verdict** (`arena.verdict`, real MultiOgar server, 15 game-min, 2x food, 8 server bots, the whole field incl. the nets in every match, 348 fresh seeds; runs/verdict_c3_4.json + runs/verdict_c3_4_b.json):

| c3_4 vs | win rate | 95% Wilson |
|---|---|---|
| g4net | 67.5% | 62-72% |
| nnbest | 66.1% | 61-71% |
| champ | 63.8% | 59-69% |
| r2g9 | 62.4% | 57-67% |
| ramp300 | 56.9% | 52-62% |
| safe | 56.6% | 51-62% |
| ramp_r2 | 55.7% | 50.5-61% |

Every lower bound is above 50%. Caveats: the margins over the r2g9 family (ramp300, ramp_r2) and safe are small (~56%); the matches were inspected as they accumulated (repeated looks make a borderline bound slightly optimistic), and the sequential test with H1 = 60% has not decided those three.
