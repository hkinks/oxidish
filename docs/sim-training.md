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

## Farming specialist (f1)

```
uv run python -m rl.ppo_sim --run f1 --init runs/rl/e1/hof/g003_m9.pt --b 128 --p 8 --farm-frac 1 --farm-players 8     --league-frac 0 --bot-frac 0 --skip 2 --bc-farmer-iters 150 --bc-lr 1e-3 --farm-eval-every 20 --curriculum-p 0.3     --explore 0 --vf-warmup 0 --ent 0.003 --vram-gb 12
```

- **The teacher:** the scripted sim bot farms only 33 alone (it gets stuck), so the teacher is `sim.bots.farmer_actions`: the heading to the nearest pellet, ~300 mass.
- **Imitation (DAgger):** the net plays, and the farmer labels its states. It runs at lr 1e-3; at the PPO rate of 1e-4 imitation barely moves (35% accuracy after 600 steps offline).
- **Results** (2 minutes alone):

  | Net | Mass |
  |---|---|
  | f1 after imitation | **250-284** |
  | the farmer | ~300 |
  | every earlier net | 20-60 |

  PPO afterwards holds ~250-260.
- **CoordConv** (`coords` gene: x/y input channels) made no difference to imitation (62% vs 61%).

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
