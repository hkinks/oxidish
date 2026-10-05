# Training in the GPU simulator: s1, evolution, viewer

How the neural bots are trained now, all in the GPU simulator (`sim/`, see [sim.md](sim.md)).
The real server (`arena/`) is used for the showcase and for a final round of fine-tuning, not
for bulk training. It is ~15-30x slower: CPU-bound, one arena per core, websocket plumbing.

```
rl/ppo_sim.py     PPO self-play trainer on the simulator (one run = one network)
rl/evo.py         evolution over network designs; every member is trained by rl/ppo_sim.py
rl/model.py       Policy: CNN actor-critic, architecture genes (convs, width, pool, input masks, shared)
rl/nohint.py      encoder that lets sim-trained nets play on the real server (rl/policy.py picks it)
sim/tourney.py    head-to-head of nets in shared 50-player sim arenas; farm() = farming-alone check
sim/progress.py   learning curve: newest checkpoint vs fixed references, every N minutes
sim/watch.py      live browser viewer (http://127.0.0.1:8096) + sim/watch.html
```

## The trainer (`rl/ppo_sim.py`)

`B` arenas x `P` players step in lockstep on the GPU. Rollouts, advantages and PPO updates all stay on the GPU.

**Slot roles in each arena:**
- **Learner:** plays the current net. Only these slots produce training samples.
- **League:** a frozen past snapshot of this run (one every `--league-every` iterations, the last `--league-size` kept), or a hall-of-fame net (`--hof DIR`). League snapshots take half the slots in `--league-frac` of the arenas.
- **Scripted bot:** `sim.bots`. Takes half the slots in `--bot-frac` of the arenas.
- **Inactive (farming drills):** `--farm-frac` of the arenas hold only `--farm-players` learners and nothing else. This gives a clean farming signal.

**Drill arenas** (they come first; the rest are normal self-play arenas):

| Flag | Arena |
|---|---|
| `--farm-frac` | only `--farm-players` learners, nothing else: clean farming signal |
| `--hunt-frac` | `--hunt-players` learners at 400-2000 mass among small scripted bots (20-200): split-kills pay off at once |
| `--flee-frac` | `--flee-players` learners at 100-600 mass and `--flee-hunters` scripted bots at 1500-6000 that chase them |
| `--giant-frac` | a full arena of learners plus `--giants` scripted bots at 5k-20k mass (like a strong human on the real server) and `--giant-learners` learners that start that big |

Drill players respawn at their drill size, so the pressure stays on all game.

**Network** (`rl.model.Policy`): 17 moves (16 headings + stay), split and eject heads, and a separate critic. The inputs are 15 grid channels (48x48, the server's real view) plus 16 scalars:

| Channels | Meaning |
|---|---|
| food, virus, mine, prey, threat, split_threat, ejected, outside | as in rl/obs.py |
| vx, vy | velocity of every moving cell over 3 steps |
| own_merge, enemy_merge | how close pieces are to being able to merge |
| owner_mass | log of the owner's total mass vs ours, painted on every enemy piece |
| **killable** | enemy cells a split half of our largest cell could eat (mass < ours / 2.6) |
| **density** | log(1 + enemy mass) summed per pixel, so a cluster of small pieces shows its worth |

`killable` and `density` exist only in the sim encoder so far: `rl/guided.py` (real server) still makes 13 channels. A checkpoint with 13 channels loads with zero weights for the two new ones (`from_checkpoint`), so it plays exactly as before.

**Reward**, per step and per learner:
- mass change / 200;
- minus all of its mass when it dies;
- plus `--virus-reward shot,pop,popmass`: per virus it shot, per enemy that popped on a virus it shot, and the popped mass / 200;
- plus `--kill-reward cell,mass`: per enemy cell eaten, and enemy mass eaten / 200;
- plus `--explore-reward` per new 500x500 square visited in a life.

There is deliberately no reward for splitting itself. Splitting is rewarded only through what it achieves, so the net can't learn to spam splits.

**Curriculum:** `--curriculum-p` of respawns (default 0.5) start big, at 50-5000 mass (log-uniform). That gives practice at large sizes, but it also pours free mass into arenas. **Max-mass numbers from training are inflated** (one life reached 193k). Judge bots only by `sim.tourney` / `sim.progress`, where everyone starts at mass 10.

**Speed:** bf16 network passes (`--no-amp` turns them off) with minibatch 2048. The update is ~2.6x faster than fp32 with minibatch 1024. One run at B=64, P=50 does ~11k samples/s with the GPU to itself.

**Memory:** `--vram-gb` is a hard cap (torch per-process fraction). On Windows a CUDA job that outgrows VRAM silently spills into system RAM and freezes the machine; the cap makes it raise OOM instead. Encoder memory grows with the number of visible cells, so 50-player arenas need ~5.5 MB per agent. Examples: B=64 P=50 needs about 18-24 GB, and B=12 P=50 fits in 6.5 GB.

**Resuming:**

```
uv run python -m rl.ppo_sim --run s1 --init runs/rl/s1/ckpt_00320.pt --start-it 320 ...
```

This resumes with the run's saved league snapshots. `--arch JSON` builds a different architecture. If `--init`'s architecture differs, the new net first imitates it for `--distill-iters` iterations: our own states, KL to its action distributions. `--max-minutes` sets a wall-time budget, and the final checkpoint is always saved.

## The s1 run (2026-10-04)

- **Setup:** pure self-play, no champion in the loop. It started from g8 (`runs/rl/g8_048.pt`), whose FOLLOW move was dropped and whose hint inputs see zeros.
- **Arena size:** 16, then 24, then 50 players per arena.
- **Rewards:** virus `0.5,0.5,1.0`, explore 0.02, and from iteration 320 also kill `0.5,1.0` plus the two new input channels.
- **Checkpoints:** `runs/rl/s1/ckpt_*.pt`. The evolution seed is `runs/rl/s1_seed_for_e1.pt` (iteration 380).
- **Learning curve** (sim.progress, shared 50-player arenas, 2 min from mass 10): iteration 240 > 180 > 100 > 20. Average mass 49 vs 17 vs 14, and deaths per slot 0.05 vs 0.53 vs 0.88.
- **Farming check** (`sim.tourney.farm`: one player alone, no enemies, 2 min):

  | Player | Mass |
  |---|---|
  | s1 iteration 380 | 105-112 |
  | s1 iteration 100 | 18 |
  | the nearest-pellet routine | 319 |

  Taking the most likely move instead of sampling doesn't change this (92). **The nets farm at about 1/3 of a trivial routine**: in crowded arenas the per-pellet reward drowns in the noise of fights and deaths. Farming drills (`--farm-frac`) are the fix being tested.
- **Splitting stays rare (0.1-0.6% of steps):** a random split usually gets you eaten, so the net learned "don't". The kill reward, the split exploration floor (`--explore`) and the `killable`/`density` inputs aim at this, and evolution tunes them.

## Population training in one process (`rl/pbt.py`, 2026-10-05)

Replaces the generation loop of rl/evo.py (whose genome and `mutate` it reuses).

```
uv run python -m rl.pbt --name p1 --from-evo e1 --b 12 --skip 2 --vram-gb 20 --hof-extra runs/rl/e1/hof >> logs/pbt-p1.txt 2>&1
```

- **One simulator for everyone:** every member owns `--b` arenas of one SimEnv, with its own drills and reward weights (`SimEnv.set_reward` is per arena), and learns there with `rl.ppo_sim.Learner`.
- **Continuous evolution:** every `--eval-minutes` the members are snapshotted. A side process (`--eval-job`) runs the tournament plus the farming and 10-minute checks while training continues.
  - When the results come back, the best member goes to the hall of fame.
  - The bottom half are replaced by mutated copies of the top half. The ranking includes the last hall-of-fame net, so a member that regressed below it is replaced by a copy of the better old net.
  - Replaced members' arenas restart.
- **State and logs:** state is in `runs/rl/<name>/evo.json`, in rl/evo.py's layout, so `sim.watch --run p1` works. Logs are `logs/pbt-p1.txt` (rounds, results) and `logs/pbt-p1-<member>.txt`.

**Speed, measured on the RTX 5090:**

| Setup | Training samples/s |
|---|---|
| 4 rl/evo.py processes | ~1,600 in total |
| rl.pbt | ~11,000 |

Where the speed came from:
1. **No cuDNN re-tuning.** PPO used a leftover minibatch whose size changed every iteration, and `cudnn.benchmark` re-tuned every convolution for each new size. That cost 5 s of CPU per update against 0.4 s of GPU work. Updates now use full minibatches only.
2. **Acting as CUDA graphs** (`GraphedAct`): one launch per network call instead of dozens of small kernels. Acting was launch-bound.
3. **One simulator for all members**, and distribution argument checks off (they sync the GPU).

**Frame skip (`--skip 2`):**
- Nets decide every 2 env steps (160 ms) and hold their move in between; a split or eject fires once.
- `gamma` keeps its per-step meaning.
- The encoder scales velocities and merge timers by the decision interval.
- The skip is saved in the checkpoint, and rl.policy, sim.tourney and sim.watch hold the move accordingly.
- Farming test: nearest pellet with 16 headings gets 256 mass when deciding every step, **303** every 2 steps and 245 every 4.

**Simulator fix: map-sized players (2026-10-05).**
- **The bug:** in long episodes, the size curriculum's respawned mass (50% of learner respawns at 50-5000) piled up in one player. Its 16 cells at max size (~360k mass) covered the 7000x7000 map and ate every respawn on its first tick. Food and viruses ran out too.
- **The effect:** after ~30 minutes, most training slots were dead. Valid samples fell from 8,850 to ~1,300 per iteration, and the old evolution runs show the same symptom (about a third valid).
- **The fix:** `SimEnv(max_player_mass=100_000)` restarts an arena when a player passes 100k mass, and rl.pbt uses curriculum 0.15. With these, players stay ~100% alive in 6000-step tests.

## Farming (2026-10-05)

**Measured farming speed:** mass after 2 minutes alone, from mass 10, 32 arenas.

| Player | Mass |
|---|---|
| perfect mouse on the nearest pellet (continuous aim) | 266 |
| nearest pellet, our 16 headings, deciding every step / 2 steps / 4 steps | 256 / **303** / 245 |
| cluster routine: each heading scored by pellets ahead, `exp(-d/150) * cos^2` | **318** |
| cluster routine, lam 300, cos^8 | 303 |
| the scripted sim bot (sim.bots.scripted_actions) | 33 (gets stuck: stops eating after ~500 steps) |
| f1 after imitating the nearest-pellet routine, sampled / greedy | 252 / 233 |
| every earlier net (s1, e1 generations) | 20-60 |

**Lessons:**
- **The action space is not the limit.** 16 headings and a decision every 160 ms farm as well as continuous aim.
- **The nets were never shown how to farm.** In crowded arenas the per-pellet reward drowns in fights and deaths. Imitation (DAgger, `--bc-farmer-iters`) lifts a net from 60 to ~250-284 in ~100 iterations.
- **Imitation needs a high learning rate.** At lr 1e-4 it barely moves (offline: 35% heading accuracy after 600 steps, 61% at lr 1e-3).
  - A bug kept `--bc-farmer-iters` out of the trainer's settings, so the first f1 run never imitated at all.
- **CoordConv** (`coords` gene: x/y input channels) made no difference to imitation (62% vs 61%).
- **Plain PPO after imitation slowly loses the skill:** 284 at iteration 120 down to 228 by iteration 440, with 8 learners per arena competing for food and eating each other. So imitation is only a starting point, and farming must be selected for directly.

**f1** (the imitation run):

```
uv run python -m rl.ppo_sim --run f1 --init runs/rl/e1/hof/g003_m9.pt --b 128 --p 8 --farm-frac 1 --farm-players 8     --league-frac 0 --bot-frac 0 --skip 2 --bc-farmer-iters 150 --bc-lr 1e-3 --farm-eval-every 20 --curriculum-p 0.3     --explore 0 --vf-warmup 0 --ent 0.003 --vram-gb 12
```

Its best checkpoint, iteration 120, is saved as `runs/rl/f1_seed.pt`.

**fp1, the farming population** (`rl.pbt --objective farm`), runs from that seed:

```
uv run python -m rl.pbt --name fp1 --objective farm --seed runs/rl/f1_seed.pt --pop 6 --b 64 --p 4 --skip 2     --eval-minutes 10 --vram-gb 20 >> logs/pbt-fp1.txt 2>&1
```

- **Arenas and reward:** farming arenas only (4 learners each) and mass as the only reward.
- **Exploration stays on:** the split/eject floor from 70 mass, and entropy.
- **Fitness is the free farming check** (`sim.tourney.farm(free=True)`), where split and W are allowed, because smaller cells move faster: the nets may find faster farming than any routine.
- **Mutations** (`rl.evo.FARM_HP_KINDS`): lr, entropy, gamma, exploration floor, explore reward, epochs, and architecture.
- **Stopping rule:** it runs until the best free-farming score stops improving.

### Farming tests, split-farming and imitation (2026-10-05, later)

**The farming test** (`sim.tourney.farm`): one player alone on the map plays 1,500 steps (2 minutes of game time) from a snapshot of the net, averaged over 16-64 arenas.
- "300" means growing from 10 to ~300 mass in those 2 minutes. It's a skill test, not a training total.
- `free=True` allows split and W. `start=1000, gain=True` measures the mass gained in 2 minutes starting from 1,000 mass.
- `--farm-eval-every` runs both parts (from 10, and from 1,000) on 64 arenas.

**Routines:** the cluster-routine sweep (lam 80-200, cos^1-3) gives 283-310 from 10, so every routine tops out at ~300.

**Split-farming only pays for big cells** (scripted: the cluster routine plus "split when the largest cell is >= X, up to N cells"):

| Start | No split | Best split variant |
|---|---|---|
| 10 | +296 | +197 (splitting costs a third) |
| 300 | +311 | +284 |
| 1000 | +374 | **+467** (split at >= 300, up to 16 cells: +25%) |

So a test from 10 mass can never reward split-farming, and a net never experiences it paying unless it trains at big sizes. fe3 trains solo with half of its respawns at 50-3000 mass.

**Imitation vs pure RL:** the same entity net, alone on the map, mass reward only.

| Net | Farming score from 10 |
|---|---|
| fe2, pure RL from scratch | 117 after 20 iterations, then **stalls at ~92-117** |
| fe1, imitating the nearest-pellet farmer | ~270 within 20 iterations |

Random discovery doesn't just take longer: it settles into a much worse habit.

**The imitation rule now:** imitation covers only the move head (the teacher never splits, and copying that would train against splitting). It fades out linearly into the plain mass reward (`--bc-fade-iters`). Split and W are left to the reward to discover.

**fe1's setup didn't work:** with 4 players per arena, PPO swung between 112 and 251 between checks, because they eat each other and blur the farming signal. Solo arenas (P=1) are the clean setup.

### fp2: the entity farming population, and virus eating discovered (2026-10-05, night)

**Setup:** `rl.pbt --objective farm`, entity nets, solo arenas.
- Fitness is both farming tests relative to their routines.
- Selection runs every 10 minutes.
- Seeded from fe3 (imitation, then PPO, ~270 from 10 mass).

**Fixes that made it work, in order of discovery:**
1. **PPO log-ratio bounded.** KL was infinite from iteration 1 for entity nets, so every update stopped after one pass. One extreme ratio wrecked fe3: 270 fell to 64 after imitation ended.
2. **Move exploration floor (2%).** Imitation had made the move choice deterministic (p = 1 - 2e-7), so a single step could swing a probability by e^20.
3. **Staggered per-arena restarts** (`--arena-steps 3000`) and farm start sizes 10-3000. Found by the research agent.
   - Solo farmers never die, so they never respawned. The size curriculum never applied, and all lives ran the whole 27-minute episode in lockstep.
   - Those lives stalled at ~270 mass, 88% of the time stuck. Most samples were noise.
4. **Cut-off lives bootstrap** from V(s_t) instead of counting as deaths, and the early stop uses the clipped share instead of an outlier-driven KL.
5. **An own cell at the centre is no pointer target:** "head toward myself" meant standing still.
6. **Velocity memory reset** on arena restarts, and **dual-clip PPO**: one floor-rare sample could steer a whole update, and m1 froze that way.
7. **Commitment inputs:** the last pointer target is flagged as a token feature, and the last direction and stay go in scalars 8-10.
8. **Batched evaluation** (`sim.tourney.farm_batch`): the one-net-at-a-time evaluation never finished a round on a busy GPU.

**Round 0 results (2 minutes alone; split and W allowed):**

| Member | From 10 | Gain from 1000 |
|---|---|---|
| **m4** (d 256, lr 2e-4) | 267 | **+910** |
| m2 | 280 | +267 |
| m0 | 263 | +281 |
| m5 | 270 | +266 |
| routines | ~300 | +374 plain / +467 split-farming |

**m4 discovered virus eating, not pellet split-farming.** Measured from 1000 mass over 2 minutes:
- it eats 8.6 viruses per player (m2 eats 2.2);
- that is ~860 of its +930 gain;
- pellets add only ~+71 net of decay.

Its many cells (~12 on average) come from virus pops; it chooses to split 0 times per minute. In the viewer, the "shaking" of a split m4 is its popped cells jostling: they can't merge for ~30 s, the server pushes them apart, and they all steer to one point. It is not indecision: there were 0% direction reversals between decisions.

**What made it discoverable:**
- **Big cells existed in training at last** (fix 3). Only a cell well above virus size can eat one, and before the fix no solo farmer got past ~270.
- **The entity input lists every virus as a token** with its mass and a "we can eat it" flag, and the pointer makes "go to that virus" a single choice.
- **The reward is mass however gained**, and virus mass counts.
- **Evolution kept the lucky variant.** Only m4 found it: the wider transformer and the doubled lr. Selection copied it at once (m6 = m4 + ent 0.002). Which gene mattered is not yet known.
- **Stable training** (fixes 1, 4, 6) meant one bad update didn't wipe it out.

**Caveats:**
- It is a real MultiOgar mechanic (eating a virus adds its mass, then pops the cell), so it works on our server.
- Popped cells are easy prey, so it is only safe without enemies. The farming fitness rewards it fully, and a separate pellet-only measure is needed to judge pellet farming.
- Virus supply limits it: a virus respawns at a rate of one per spawn event while below the minimum count.

## Evolution (`rl/evo.py`)

Weights are learned by PPO. Evolution searches over the **design**.

**Genome:**
- Architecture:
  - conv layers (channels 16-128, stride 1/2, 2-6 layers);
  - width 128-512;
  - input pooling 1/2;
  - which input channels the net sees;
  - shared or separate critic trunk.
- Training settings:
  - lr, entropy, gamma, epochs;
  - split/eject exploration floor;
  - kill, virus and explore reward weights;
  - farm-drill share.

**One generation:**
1. Every member trains for `--minutes`, all at once in their own `rl.ppo_sim` processes (`runs/rl/<name>/<member>/`, `logs/evo-<name>-<member>.txt`). Opponents are the member's own past snapshots plus the hall of fame.
2. Members play each other in shared 50-player arenas (`sim.tourney`, 12 arenas x 4 min). Fitness is the mean log of average mass. The newest hall-of-fame entry plays too, as a fixed yardstick.
3. Two further checks are reported:
   - farming alone (`farm`);
   - a 10-min game from mass 10, as an honest "can it get big" number.
4. The winner is copied to `runs/rl/<name>/hof/`. The bottom half is replaced by mutated copies of the top half:
   - same architecture: the child inherits the weights;
   - new architecture: the child imitates its parent first, and gets one generation of grace from elimination.
5. State goes to `runs/rl/<name>/evo.json`, and the loop resumes from it after a restart. If a member crashes, it continues from its last checkpoint.

```
uv run python -m rl.evo --name e1 --seed runs/rl/s1_seed_for_e1.pt --pop 4 --minutes 30 \
    --b 12 --p 50 --vram-gb 6.5 >> logs/evo-e1.txt 2>&1
```

TensorBoard shows each member as `e1/m<k>`.

## Watching

- **Viewer:**

  ```
  uv run python -m sim.watch --run e1      # or --run s1
  ```

  It opens on http://127.0.0.1:8096.
  - **What it runs:** its own live sim arena of 50 players (not a replay of training games).
    - Evolution runs: every current member plus the newest hall-of-fame net, in equal shares.
    - Plain runs: the latest net plus 3 past snapshots.
    - Either way, 6 scripted bots join.
  - **Rounds:** models change only between rounds. When a generation finishes (or a plain run saves a checkpoint), the "new round" button turns green. Pressing it loads the new models and restarts with everyone at mass 10.
  - **Panels:** cells are coloured by model. The Models panel shows avg/best mass per model, and the player list is ranked by mass (click to follow, again to stop). The progress chart shows the best/median member and the hall of fame per generation, or `progress.json` for plain runs.
  - **Speed:** it runs at real time. Faster isn't possible while training holds the GPU: a step costs ~45 ms of waiting behind training's kernels whatever the arena size. It stops simulating when no page has been open for 5 s.
- **TensorBoard** (`logs/tb`, http://127.0.0.1:6006):
  - training runs: `play/*` (life stats, kills, virus shots/pops) and `train/*`;
  - `<run>-eval` from sim.progress: `models/avg_mass|deaths|kills` per model, and `eval/margin_vs_best_ref`.
  - Old real-server runs g3-g11 are in `logs/tb_archive`.
- **Learning curve:**

  ```
  uv run python -m sim.progress --run s1 --refs <ckpt> <ckpt> --backfill 2
  ```

## Lessons

- The sim tournament agrees with the iteration order (240 > 180 > 100), so it is usable as fitness. Still, sim results need a real-server check before they're believed (see the winner's-curse note in training-plan.md).
- The viewer at "max" speed halved training throughput. Small GPU jobs next to training cost far more in waiting than in work.
- Rule changes as fixes for learned behaviour are a dead end:
  - Big-cell hunting in the champion (`big_hunt`, bot/params.py) **lost** 33-67 to plain farming in 24 long 2x-food matches. It is now off by default.
  - Strategy (hunt when big, split-kill, farm efficiently) has to come from learning. The rules are only useful as sparring partners.

## Ideas from other agar.io RL projects (2026-10-04)

Surveyed GoBigger (OpenDILab, ICLR 2023), AgarCL (arXiv 2505.18347), the RPG paper (arXiv 2103.04564, staghuntrpg/agar), Wiehe et al. 2018 and AgarAI.
- **Nobody has solved it.** AgarCL's DQN/PPO/SAC (160M frames each) never learned useful splits nor beat the fixed bots. Wiehe et al. beat a greedy bot at pellet collection, but not at fighting. RPG got splits against scripted cells only with reward randomization plus fine-tuning.
- **No usable pretrained agents.** Only RPG ships checkpoints, for its own 2-player environment.
- **Ideas worth trying here, ranked:**
  1. Decide every 4-8 ticks (all three projects use frame skip/action repeat), and make a split a committed macro (hold the heading for a few steps), so its payoff isn't lost in per-step credit.
  2. Explore splits only where they can pay: sample the split floor only when a `killable` target is in split range.
  3. Opponent curriculum: ramp scripted-bot speed/aggression from 0 to 1 early (RPG), with a ladder of food-collector, then aggressive, then evasive bots (GoBigger).
  4. Reward randomization: per-agent random weights on kill bonus, loss penalty, split bonus and explore bonus (RPG). A no-loss-penalty variant pushes hunting.
  5. Fix farming by imitating the nearest-pellet routine first (behaviour cloning, then PPO, or a KL term).
  6. Steadier continued training: PPO clip 0.1, a KL stop, and a gate against the best old net.

## 2026-10-05 (evening): real-server baseline, MultiOgar bots in the sim, the ffa population

**Real-server baseline** (`arena.tournament`, 24 fresh-seed matches, 4 game-min, 2x food, 8 MultiOgar bots;
`runs/bench/base_50000.json`, `logs/bench-base.txt`):

| bot | avg mass | deaths/match | beats gen |
|---|---|---|---|
| r2g9 | 306 | 0.85 | 88% |
| champ | 294 | 0.94 | 79% |
| g4net | 264 | 1.21 | 79% |
| nnbest | 227 | 1.29 | 79% |
| evader (v1, sim-trained) | 202 | 2.67 | 67% |
| gen (g1 round-2 best, sim-trained) | 162 | 5.15 | - |

**Why the sim nets lose** (the tournament now records who eats each of our cells: `print_losses`):
- gen and the evader lose 43% / 51% of their eaten mass to **MultiOgar's own bots**, the weakest players there; champ loses very little to them.
- The eaten cells are small split pieces (~30 mass): 74% were eaten while we were split, and 85% by a split killer.
- MultiOgar's bot (`vendor/MultiOgar/src/ai/BotPlayer.js`) split-kills deterministically: any cell it out-sizes 1.69x in radius (and over 0.4x its radius) within split reach, then chases it for 20 ticks.
- The sim's scripted bot split at random on the nearest prey only. g1's layout had no free-for-all bot arenas at all (`bot_frac` 0), and the split "fun" bonus paid for splitting.

**g1 round 3** (human imitation on, weight 0.5): every member fell from fitness ~-0.3 to ~-1.7, and hunting went negative. g1 was stopped.

**Changes:**
- `sim.bots.MultiOgarBots`: the real server's bot, vectorised; `SimEnv(bot_kind="multiogar")`, shared through `env.bot_actions()`.
- `rl.pbt --objective ffa`:
  - training: 40% free-for-all arenas (half learners, half MultiOgar bots), 25% league arenas, and farm/hunt/flee drills; no split bonus, no imitation; split exploration only from 150 mass; kickstarting from the specialists as in general.
  - **fitness: a real MultiOgar tournament each round** (`ffa_eval`: 24 fresh-seed matches, members x2 with champ and r2g9), as the log-mass margin over the rule-based champions in the same matches. It runs on the CPU while training continues.
- `arena.tournament`: `play_many`, `--out`, and the loss report.

Run: see `ffa1` (pop 4, b 32, `--vram-gb 18`). Keep total GPU memory under ~26 GB: the 0x133 crash happened with VRAM nearly full.

**ffa1 round 0 diagnosis** (real server, ffa1 round-0 best m2; `runs/bench/greedy_61000.json`, `nosplit_62000.json`):
- **Greedy play is worse** than sampled (log-mass 4.95 vs 5.18), so sampled play stays.
- **The net never splits** (0 splits in 3000 steps); masking split / W off changes nothing (5.02).
- **Its losses are virus pops:** it loses 452 cells while holding 8+ cells, where champ loses 24. It pops 0.6x per 2 game-min; champ never does. The server's bots eat the pieces.
- **Cause:** solo farm drills and the farm teacher (fp2) reward eating viruses, which is safe only when nobody can eat the pieces.
- **Fix:** the ffa layout has no farm drills and no farm teacher (bot arenas 50%). The real-server matches now count own pops (`AgentStats.popped`), reported each round. A `greedy:` / `nosplit:` lineup prefix is available for diagnostics.

**ffa1 round 1:** the best member's margin went from -0.69 to -0.59, but pops stayed at ~1.2 per 4-min match (champ 0.2, r2g9 0.25). With MultiOgar bots, the sim reproduces the real numbers: m2 pops 0.46/min and dies 3.2x per 4 min in the sim, vs 0.3/min and 3.6x on the real server. So `sim.tourney.run(..., bot_kind="multiogar")` (now with `pops_per_min`) is a usable fast proxy.

**Pop penalty:** `SimEnv.set_reward(pop_penalty=...)`. Each own virus pop costs `pop_penalty x our mass before it` in reward units (linear, /200), so the cost of the pieces arrives with the pop. ffa uses 0.5: a 400-mass cell that pops nets -0.5 instead of +0.5. Own pops are counted in `World.popped` (part of the CUDA-graph state).

**Play-time guards** (`rl/policy.py`, lineup prefixes `guard:` and `guard2:`; real server, ffa1 round-1 m5, 24 matches each):

| variant | log-mass | deaths/match | pops/match |
|---|---|---|---|
| net | 4.83 | 3.06 | 1.04 |
| + virus guard (champion's detour) | 5.06 | 3.42 | 0.12 |
| + threat guard (champion's danger radius push) | 5.07 (guard 4.87 in the same matches) | 1.19 | - |
| champ | 5.56-5.61 | 0.69-0.94 | 0.38 |

- With the virus guard on, most remaining deaths are cells under 30 mass, right after spawning, split-killed by MultiOgar bots (90 of 164). Champ has 9 such deaths.
- The threat guard brings deaths close to champ's. The remaining gap is growth (avg 192 vs 314, peak 543 vs 863).
- ffa's real-server evaluation now plays the members with both guards (`guard2:`), the way they will be deployed.
- **Training under the guards:** `sim.bots.guard_mouse` is the same two guards vectorised. They give identical headings to `rl.policy`'s on 64 random scenes. ffa sets `guard=2`, so learners and league nets move under them, and the net learns what pays within them (growth, hunting).

**The growth gap is farming without virus pops** (real server, alone, 2 game-min, 8 seeds, peak mass): net 258, + virus guard 187, + both guards 174, champ 240, farmM4 214. The nets farm partly by popping on viruses, which only pays when nobody can eat the pieces.

**Imitating the champions failed** (`rl/champ_data.py`: 72k real-server frames of champ / r2g9 in the entity format; `rl/champ_bc.py`: move + split likelihood with a KL anchor). Held-out heading accuracy reached 90% (from 0%: the net moves by pointer only). On the real server it lost badly (log-mass 4.49 vs 5.18 for the net it started from; deaths 2.5 vs 1.2). The champion's precise steering squeezed into 16 headings, plus compounding errors, does not make a player.

**Round 5: plateau at about -0.3, and the cause is growth.** The guarded nets now lose *less* mass per match than champ (116-145 vs 291) and kill as often (7.2), but they stay smaller. Farming under the guards, the sim matches the real server (g002_m3: sim 179, real 174-187):

| farmer (2 min alone, guarded) | mass |
|---|---|
| ffa nets | 168-179 |
| nearest-pellet farmer (`sim.bots.farmer_actions`) | 222 |
| champ (real) | 240 |

**Fix:**
- The ffa layout has solo farm drills again (20%, 2 farmers each), now under the guards, so virus eating is impossible.
- There, `farm_imitate` pulls the move toward the farmer's nearest pellet (as a pointer), only on farm-drill samples, fading from 1.0 to 0.1 over 300 iterations.
- `sim.tourney.farm_batch(guard=2)` is the fast check.
