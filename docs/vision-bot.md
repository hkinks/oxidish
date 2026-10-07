# Vision bot: plan (2026-10-07)

A bot that decides from **rendered frames**: the Rust server (`rust/ogar`) draws each controlled player's camera
view as a small image, and a CNN policy picks the move from the pixels. Actions still go through the normal agent
interface (heading, split, eject). There is no screen capture: frames are rendered headless inside the engine,
so training scales to many worlds.

**Goal.** Beat the planner (≈40% leader share; c3_4 ≈30%, mean of 5 seed sets) on the endless metric, toward the
80% goal. The frames carry only what the client shows: the visible cells, their colours and sizes, and the border.
Hidden state and other players' internals are not included.

**Honest prior.** Our earlier neural bots (grid CNN in `rl/model.py`, trained with PPO in `sim/`) never matched
the rule bots, and never learned splitting (see [architecture-research.md](architecture-research.md); AgarCL hit
the same wall). Pixels are a harder input than our hand-made grid. So this plan does not start from RL. It starts
by **imitating the planner on frames**, and uses RL only to improve on that base. Every stage has a gate, and the
project stops at the first gate it misses.

## Architecture

```
rust/ogar (world, rule-bot agents)
  └─ render.rs   Snapshot -> uint8 frame [C,H,W]      (same data the client gets; no privileged info)
rust/ogar_py (new, PyO3 + maturin)   VecEnv: N worlds in-process, threads over worlds
  reset() / step(actions) -> frames [N,C,H,W] uint8 (numpy, no copy), reward, done, info
  teacher actions: the planner's choice for the same step (for imitation / DAgger)
Python (ML only, per rust-first)   rl/vision/: model, BC, PPO, export to ONNX
rust/ogar agent kind "vision"   runs the ONNX net in-process (crate `ort` or `tract`)
  -> arena.endless / verdict / live servers work unchanged
```

**Why render from `agent::snapshot`.** It is the exact view the protocol sends a player, so a frame shows no
more than a human sees. The renderer is a pure function of the Snapshot, which makes it unit-testable and lets it
reuse the client's camera maths (`rust/game_client`: centre, zoom = server's `scale`).

## Design choices (recommendations; easy to change later)

- **Frame.** 2 views stacked as channels:
  - a **96×96 local view** of the server's camera rectangle;
  - a **48×48 wide view** at 2× that extent.
  - **Why two:** at big mass, pellets are under one pixel in a single view, and threats arrive from off-screen.
- **Plain vision, grayscale.** Draw one grayscale channel per view. Sizes come only from how big a circle looks on
  screen; there is no hand-made feature channel such as size, mass or a threat flag, which would be cheating
  (user, 2026-10-07).
  - **Cells:** filled discs with an outline.
  - **Viruses:** a spiky outline, recognisable by shape.
  - **Pellets and ejected mass:** small dots.
  - **Outside the map border:** dark.
  - **Our own cells:** their own brightness. This stands in for a human seeing their own name.
  - **Other players:** one gray shade per player, which keeps the client's colour cue for which pieces belong to
    the same player.
- **Motion.** Stack **2 frames** (now and 4 ticks ago). Velocities matter, and a stack is simpler than a recurrent
  network (GRU).
- **Action.** A **spatial action map**: the net outputs a heat map over the local view, and the argmax pixel is
  the cursor target ("go here"; AlphaStar-style). It also outputs split and eject logits. A target pixel matches
  the visual input better than 16 fixed headings, and the planner's action converts to a target pixel exactly.
  Fallback: the 16 headings plus stay that we already proved work (303 vs 266 farming).
- **Decision rate.** Every 2 server ticks (80 ms), as the rule bots do.
- **Network.** IMPALA-ResNet (3 stages, no early stride on the local view), a heat-map head via 1×1 convs, and a
  pooled MLP for the value, split and eject heads. Small enough to run in-process on CPU for evaluation
  (aim ≤ 1 ms per decision with `ort`).

## Stages and gates

| # | stage | output | gate (stop if missed) |
|---|---|---|---|
| M0 | `render.rs` plus PNG dump (`ogar render --seed S --tick T`) and unit tests (circle sizes, camera, border) | frames that look like the client | a side-by-side PNG next to a client screenshot looks right to you |
| M1 | `ogar_py` VecEnv (PyO3), threaded worlds, rule-bot opponents in-process, teacher actions | `pip install -e rust/ogar_py` | ≥ 20k frames/s on ≤ 10 cores, CPU ≤ 90% |
| M2 | **Imitation**: behaviour cloning on planner actions, then DAgger (the student drives, the teacher labels) | `runs/vision/bc_*.pt` | target-pixel agreement ≥ 80%, split agreement ≥ 70%; ≥ 45% vs c3_4 in 15-min matches |
| M3 | ONNX export plus agent kind `"vision"` in ogar | `bot/vision_bc.json` | identical actions Python vs Rust on 1k frames; endless leader share ≥ c3_4's (~30%) |
| M4 | **PPO fine-tune** from the BC net, with a KL anchor to the teacher (decaying), against a league of rule bots, the planner and past selves | `runs/vision/ppo_*.pt` | endless leader share > planner (40%) on the mean of 5 seed sets and on ≥ 3 of 5 |
| M5 | Scale and tune (resolution, frame stack, reward shaping toward leader share) | | each change kept only if it wins on ≥ 2 seed sets |

**Reward for M4.** Δ log(own mass), plus a bonus while we are #1 on the leaderboard (the goal metric), minus a
death penalty. The value head sees the leaderboard rank as a scalar.

**Training data size.** The teacher costs ~0.3 ms per decision. With 10 cores, BC data collection gives about
10⁷ labelled frames per hour. That is enough for M2; PPO in M4 is limited by the GPU, not by the engine.

## Resources and rules

- Rust for the engine, renderer, bindings and evaluation agent; Python only for the network and training loops.
- GPU (RTX 5090): training only, VRAM ≤ 26 GB, power 400 W. The existing ffa1 RL run stays paused; this project
  is separate and starts only when approved.
- CPU ≤ 90%, so the VecEnv thread count is capped (`--cpus`). Live servers (9600/9700) stay pinned to cores 11-15.

## Open questions

1. (Settled: plain grayscale vision, no feature channels.)
2. Teacher: the **planner** (best, ~0.3 ms) or c3_4 (simpler, faster labels)? The plan uses the planner, with c3_4
   as fallback.
3. Should it later run against human-style clients (screen capture)? Not planned now; the renderer could be made
   to match the client's look later if wanted.

## Effort estimate

| milestone | effort |
|---|---|
| M0 | half a day |
| M1 | 1 day |
| M2 | 1-2 days (mostly training time) |
| M3 | half a day |
| M4 | multi-day training with checkpoints in between |

Each gate is a natural point to decide whether to continue.

## M1 status: `ogar_py` VecEnv (2026-10-07, branch `worktree-agent-ad386154cc2767e6b`)

**Code.** Engine-side env in `rust/ogar/src/vision.rs` (pure Rust, unit-tested); Python bindings in `rust/ogar_py`
(PyO3 0.25 + rust-numpy, abi3-py310); renderer speed-up in `rust/ogar/src/render.rs`; tools in `rl/vision/`
(`test_env.py`, `bench.py`, `collect.py`).

**Build** (maturin is a dev dependency in `pyproject.toml`; a light venv is enough, no torch needed):

```
uv venv .venv --python 3.10 && uv pip install --python .venv/Scripts/python.exe maturin numpy pytest
# from rust/ogar_py, venv active (VIRTUAL_ENV set); -j 2 keeps the build polite
maturin develop --release -j 2
# from the repo root (specs like bot/c3_4.json are relative to the cwd)
python -m pytest rl/vision -q          # Python smoke tests;  cargo test --release (in rust/ogar) for the Rust ones
```

**API**

```python
env = ogar_py.VecEnv(n_worlds, learners_per_world, opponents=[("c34", "bot/c3_4.json", 4)],
                     teacher="bot/plan_c34.json", px=256, sync=2, bots=8, food_mult=2, map=7000,
                     seed=1, threads=8, warmup=0, label=True)
obs = env.reset()                                   # {"view": u8[N,2,px,px], "wide": u8[N,2,px/2,px/2]}
obs, reward, done, info = env.step(actions)         # actions f32[N,4] = (u, v, split, eject)
env.set_teacher(True | bool_list)                   # teacher drives (actions ignored); per learner for DAgger
env.info(); env.timing(); env.teacher_counters()
```

- `N = n_worlds * learners_per_world`. Each world = server bots + the opponent rule bots (in-process `AgentDriver`s,
  decide every `sync` ticks) + the learners (`PKind::Agent`, controlled from outside).
- Frames: channel 0 = this decision, channel 1 = the previous decision (2-frame stack; a new life starts with both
  equal). `view` = camera square (centre = view centre, half-width = view-box half-width), `wide` = 2x extent at px/2.
  Arrays are zero-copy views of the env's buffers (valid until the next step/reset; `copy=True` for private copies).
- Action: target point in the **view frame's pixel coordinates** (continuous, 0..px), mapped to world through the
  camera of the observation it was chosen from; split and eject flags (> 0.5).
- `info["teacher"]` f32[N,4]: the teacher's decision **for the current observation**, in the same format (BC label).
  The rule bots' `Out` is a world-space cursor target; it is converted with the exact camera inverse and, when it is
  outside the frame, pulled in along the ray from the frame centre (heading preserved). A Rust test checks that
  teacher target -> pixel -> world lands within 1 px (it is exact up to float).
- `info`: `mass`, `rank` (0 = largest player in the world, exact, computed every decision), `leader` (rank 0), `ncells`.
  `reward` = change of own total mass, a death gives `-mass_before`. On death the learner respawns at once
  (`done = 1`, the returned obs is the new life's first frame). Worlds never end; `reset()` builds fresh worlds
  (new seeds per call). Log-mass reward / leader bonus are computed in Python from `mass` / `leader`.
- Worlds step in parallel on `threads` scoped worker threads; the GIL is released during `step`.
- The teacher runs on every decision (state-keeping and labels) even when the student drives (DAgger). `label=False`
  skips it for pure evaluation.

**Renderer optimisation** (look kept: before/after PNGs in `logs/vision/before` and `logs/vision/after`, RMSE 0.3% on
a 256 px frame): `render_into` works in u8 with a reusable scratch; background/border by row spans; pixels well inside
a disc take the fill directly; edge pixels use distance-based anti-aliasing instead of 4x4 supersampling; viruses use
a polynomial `cos(18 theta)` instead of `atan2` per sample; sub-1.6 px discs (pellets) are one coverage-weighted dot.
256 px frame: 187 us -> 22 us, 128 px frame: 29 us -> 4 us (`cargo run --release --example render_bench`).

**Benchmark** (32 worlds x 4 learners = 128 learners, px 256, 4 `c3_4` opponents + 8 server bots per world, teacher-driven,
learner decisions/s including world ticks, opponents, render and teacher; 8 physical / 16 logical cores, machine ~50% busy
otherwise; 1500 warm-up ticks):

| teacher | threads 4 | threads 8 | threads 10 | gate 20k at 10 |
|---|---|---|---|---|
| planner (`plan_c34.json`) | 9.2k | 13.6k | 14.8k (15.9k after 5 min of game; 17.5k in a 600-step run) | **missed** |
| c3_4 rule bot | 18.6k | 31.2k | 35.3k | met |
| none (`label=False`, student evaluation) | | | 54k | met |
| planner, renderer before optimisation | 7.4k | 12.6k | 14.5k (render was 51% of the cost) | |

CPU per decision at 10 threads (planner teacher): world ticks 30 us, opponents 84 us (4 per world), snapshot 1 us,
render 36 us (2 frames), teacher 265 us. So the **planner is 64% of the cost** (c3_4 base rule ~70 us + deep eval
~155-195 us). The plan assumed the deep eval fires on ~6% of steps; measured, the gate passes on **61-67%** of steps at
0.25-0.29 ms each. Without optimising the planner itself (out of scope, behaviour must not change) the options are:
planner labels for DAgger / evaluation sets only and `c3_4` labels for bulk BC (35k/s), fewer opponents per world
(2 opponents: 42 us instead of 84), or accept ~15-17k/s for planner data (about 5.5e7 labelled frames per hour,
which is still above the 1e7 per hour M2 needs). Pure student rollouts (M4) run at 54k decisions/s before the network.

**Data collector** `rl/vision/collect.py`: time-major shards (`view`, `wide`, `teacher`, `action`, `reward`, `done`,
`mass`, `rank`, `leader`, `first`; current frame only, the previous frame of the stack is the previous row unless
`first`). Raw 82 KB per decision (about 3 TB/h at full speed); with `--compress` (deflate level 1) **2.3 KB per decision,
about 50 GB/h at 6k decisions/s end to end**; zlib, not the engine, then limits collection (about 0.7 ms CPU per
decision), so for M2 prefer streaming straight from the env into the trainer (labels come with every step) and keep
shards only for fixed validation sets. `--noise S` executes teacher + Gaussian pixel noise (DART-style) while the
label stays the clean teacher action.

## M2 status: imitation + DAgger (2026-10-07/08, branch `worktree-agent-a01a58030cb44bc32`)

**Result in one line.** Target-pixel agreement gate met (80.8% on the teacher-driven held-out set, early BC/DAgger
checkpoint); **split-agreement gate missed** (best F1 about 0.3); in-env, the student **without splitting matches the
planner's leader share** (0.215 vs 0.207, 33-game-minute runs, 2 seed sets) but dies more often (14-15 vs 8 per hour).
Splitting students are worse than not splitting. The "45% vs c3_4 in 15-min matches" part of the gate needs the M3
in-server agent and was not measured.

**Code** (`rl/vision/`, Python/PyTorch, no Rust change): `model.py` (VisionNet), `store.py` (GPU ring), `train_bc.py`
(streaming BC + DAgger), `make_val.py` (fixed validation sets), `evalval.py` (held-out metrics), `eval_env.py` (in-env
student/teacher runs), `export.py` (TorchScript + ONNX), `summ.py`, `bench_model.py`.

**Model.** 1.49M parameters. View trunk: IMPALA stages 2->16 (3x3 conv at the full 256 px, then 2x2 max pool, 1 res
block), 32 (2 res) at 128->64 px, 64 (2 res) at 32 px, 128 (1 res) at 16 px. Wide trunk (128 px, 3 light stages, 64 ch
at 16x16) fused at 16x16. Heat-map decoder (U-Net style, skips from the 32 and 64 px stages) gives 64x64 logits: one
cell = 4 view px; target = argmax cell refined by a 3x3 soft-argmax. Split / eject / value come from an MLP on the
pooled features plus an 8-channel 16x16 spatial summary. bf16 autocast, channels_last; training 5.6k samples/s alone,
3.5-4.5k/s with the env and student inference running. Value head is built but untrained in M2.

**Training** (`python rl/vision/train_bc.py --name bc2 --hours 2 --d0 0.05 --d1 0.4 --pmax 0.9 --ring 320 --pos-weight 3`;
the validation sets first: `python rl/vision/make_val.py` and `make_val.py --events 600 --out runs/vision/val_events
--seed 888000 --worlds 32`). 64 worlds x 4 learners, opponent field = champ, r2g9, safe, ramp_r2, ramp300 + 8 server
bots, planner teacher, 10 env threads pinned to cores 0-10. Collector thread steps the env and pushes frames + teacher
labels into a GPU ring (320 decisions x 256 learners = 82k frames, 6.7 GB); the trainer samples it uniformly
(batch 256, AdamW, lr 6e-4 cosine, EMA 0.999, loss = cross-entropy on a Gaussian (1 cell) heat-map target + BCE for
split / eject). Dihedral augmentation (8 flips/transposes with matching labels). Collection is throttled to 1 new
sample per trained sample. DAgger: each learner is driven by the teacher or the student in segments of ~150 decisions;
the student fraction ramps linearly to `pmax` over the budget fractions d0..d1; labels are always the planner's.
Split is rare (0.07% of decisions, about 30 per game hour): every teacher-split frame goes to a FIFO store (16k frames)
and 3-6% of each batch is drawn from it. VRAM about 22-23 GB total on the card (incl. 2.4 GB of other processes), 400 W.
Total training wall time about 4 h (bc1 25 min prototype, bc2 2 h, bc3 1.5 h continuation); about 54M samples.

**Held-out metrics** (51,200 teacher-driven frames from 1024 learner-windows, seeds 777000+, game minutes 3-16; plus
600 mined teacher-split frames). Agreement = angle between student and teacher heading from the frame centre < 15 deg.

| checkpoint | student share of drivers | target agree (<15 deg) | <30 deg | median px err | split P / R / F1 (best thr) | split recall on 600 events |
|---|---|---|---|---|---|---|
| bc1 10 min (BC only, 2.8M samples) | 2% | 0.791 | 0.855 | 2.6 | 0.08 / 0.19 / 0.11 | 0.17 |
| bc2 10 min (`runs/vision/bc2/best.pt`) | 9% | **0.808** | 0.868 | 2.3 | 0.27 / 0.17 / 0.21 | 0.14 |
| bc2 final (2 h, DAgger to 90%) | 90% | 0.680 | 0.779 | 5.7 | 0.32 / 0.17 / 0.22 | 0.21 |
| bc3 final (continued, 70% student) | 70% | 0.768 | 0.834 | 3.3 | 0.47 / 0.19 / 0.27 | 0.14 |

Agreement on the teacher-driven set falls as the DAgger share rises because the ring then holds mostly student-visited
states (a shift in the training distribution, not worse play: in-env results improve, see below). Split agreement
stays far below the 70% gate: precision and recall both about 0.2-0.5 at the best threshold. Eject: the planner never
ejects in this field (0 positives), so the eject head is untested.

**In-env** (32 worlds x 2 learners per world, same field, learners driven purely by the policy; leader = largest
player in the world; deaths per learner-hour of game time). Seed 424242, 16 game min:

| driver | mean mass | leader share | top-3 | deaths/h |
|---|---|---|---|---|
| teacher (planner) | 1663 | 0.207 | 0.467 | 8.4 |
| bc1 (80% agree, 22% DAgger) | 317 | 0.008 | 0.083 | 56.7 |
| bc2 @1 h (52% DAgger) | 1387 | 0.150 | 0.404 | 21.6 |
| bc2 final, splits on (thr 0.85) | 1450 | 0.146 | 0.421 | 15.7 |
| bc2 final, **no split** | 1816 | 0.233 | 0.540 | 10.3 |

Seeds 11 and 12, 33 game min each (mean of the two): 

| driver | mean mass | leader share | top-3 | deaths/h | splits/h |
|---|---|---|---|---|---|
| teacher (planner) | 2402 | 0.207 (0.214 / 0.199) | 0.52 | 8.0 | about 30 |
| bc2 final, no split | 2427 | 0.214 (0.224 / 0.204) | 0.50 | 13.6 | 0 |
| bc2 final, split thr 0.97 | 2079 | 0.178 (0.178 / 0.178) | 0.45 | 16.5 | 10 |
| bc3 final, no split | 2340 | 0.216 (0.226 / 0.206) | 0.48 | 15.5 | 0 |
| bc3 final, split thr 0.90 | 2332 | 0.197 (0.209 / 0.184) | 0.48 | 14.6 | 20 |

Leader share differences of about 0.02 are within noise (per-world sd 0.12 over 32 worlds). Read: pure DAgger-trained
pixels reach planner-level leader share and mass, but die about 1.7x as often, and splitting hurts.

**What limited it.**
1. Split: the label is 0.07% of frames and depends on sub-pixel size ratios (prey > 1.3x, within split range) and on
   hidden state (split cooldown). Oversampling positives raised precision a little (0.1 -> 0.3-0.5) but recall stays
   about 0.2. Student splits are mostly wrong-time splits that cost mass; the shipped recommendation is no split
   (`split_thr` stays in the export json at 0.9 for experiments).
2. Distribution shift: BC alone at 80% agreement collapses in-env (mass 317, 57 deaths/h); the DAgger share is what
   matters, and the in-env numbers plateaued between bc2 and bc3 (extra 1.5 h at lower lr gave the same leader share).
3. Deaths/hour 14-15 vs 8: single bad decisions near threats; ring memory is only 320 decisions per learner (about
   26 game seconds), so rare dangerous situations are forgotten. A larger or prioritised replay is the first thing to try.
4. Throughput: the trainer, not the env, is the bottleneck (about 4k samples/s of the 15k/s the env can label).

**Checkpoints and export** (`runs/vision/` is gitignored, nothing large is committed): `runs/vision/bc3/final.pt`
(recommended, play with `--no-split` semantics), `runs/vision/bc2/final.pt`, `runs/vision/bc2/best.pt` (80.8% agreement),
logs in `logs/vision/*.jsonl`. M3 export: `runs/vision/export/vision_bc3.{pt,onnx,json}` (TorchScript / ONNX,
inputs u8 view[B,2,256,256], wide[B,2,128,128]; outputs heat[B,64,64], split, eject, value; ONNX vs torch max
abs diff 0.04 on random noise input, TorchScript exact). Decode as in `model.decode_target`; the planner-pixel frame is
the view frame. Reproduce the export: `python rl/vision/export.py runs/vision/bc3/final.pt runs/vision/export/vision_bc3`.
