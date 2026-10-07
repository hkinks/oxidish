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
