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

## M3 status (2026-10-08): the vision bot in the Rust server

**Spec:** `bot/vision_bc3.json` = `{"kind":"vision","model":"runs/vision/export/vision_bc3.onnx","split":false}`.
Optional keys: `split`, `split_thr`, `ep` (`cuda` = CUDA graphs, default; `cuda_plain`; `cpu`), `eject`, `threads`.
It works wherever a genome file does: `ogar match`, `ogar serve --agents`, `arena.endless --agents`,
`arena.verdict --engine ogar`.

**Code and runtime**
- `rust/ogar/src/visionbot.rs`, `nn.rs`. Frames come from `vision::FrameStack`; a test checks them as byte-identical to the VecEnv
  over 6000 stacks, including deaths and respawns.
- `AgentDriver::step_all` batches all vision agents of a world into one forward pass.
- Runtime is onnxruntime-gpu 1.23.2 through `ort` (`load-dynamic`). The DLLs live in `runs/vision/ort/` (gitignored);
  cuDNN and cuBLAS come from `.venv/Lib/site-packages/torch/lib` or `OGAR_DLL_DIRS`.

**Parity with torch** (1000 frames, `ogar vdump` + `rl/vision/parity.py`): all headings within 1° (max 0.01° on
CUDA), and no split-decision mismatches.

**Cost:** about 0.9 ms per batched call alone and 4.9 ms with 8 processes sharing the GPU (CPU: about 37 ms). An endless run
(8 worlds × 240 game-min) takes 1331 s, against about 320 s for the planner.

**Leader share** (arena.endless, 8 worlds × 240 game-min, the usual field; splitting off)

| spec | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|
| vision `vision_bc3` | **48** | **36** | **40** | **42** | 44 | **42.0** |
| planner v1 `plan_c34` (its teacher) | 46 | 35 | 38 | 38 | 44 | 40.2 |
| c3_4 | ~31 | 28 | 31 | | | ~30 |

The M3 gate (≥ c3_4) is met. The vision bot beats its teacher on 4 of 5 seed sets and ties the fifth. The margin
is small (+1.8), but it was consistent across all five sets. It still dies about 1.7× as often as the planner
(M2 in-env), which is the first thing for M4 to fix.

## Fast endless evaluator (`ogar endless`)

`ogar endless --agents name=spec:N ... --worlds 8 --game-min 240 --seed-base S [--threads T] [--out samples.json]` runs all
worlds in ONE process (rust/ogar/src/endless.rs), headless and unthrottled. Each world is built exactly like
`ogar serve --seed S+i --agents ...` (same `World::new`, agent order and names; tick, then a decision step every
2nd tick, then the 15 s `[players]` sample), so a rule-bot lineup gives the same samples as `arena.endless`
(**verified bit-identical**: 2 worlds x 15 min and 30 min, c3_4:2 + champ_safe:2, every sample equal). Rule bots
step on worker threads; for vision lineups the workers render frames, a coordinator runs ONE batched net call for the
vision agents of all worlds (16 frames for 8 worlds x 2; graph sizes are padded to powers of two inside `infer_batch`)
and the workers apply the actions (two barriers per decision).

Python: `uv run python -m arena.endless ... --engine inproc [--shards K] [--threads T]`; same output lines and `--out`
json as the default `--engine serve` (the engine's own timing goes to stderr). `--shards K` splits the worlds over K
`ogar endless` processes: on a GPU shared with other jobs every process gets its own time slice, which helps when
the card is busy.

**Equivalence.** Rule bots: bit-identical (above). Vision: with one world (batch 2, same as `serve`) the samples are also
bit-identical (3 min); with larger batches the GPU kernels round differently, so worlds diverge chaotically from the
first minutes (not a logic difference). Leader share, `v=vision_bc3:2 champ_safe:2`, 8 worlds x 60 min: seeds 700
serve 69% / inproc 56%; seeds 900 serve 55% / inproc 70%; means 62% vs 63%, i.e. inside seed noise.

**Cost.** The batched call is GPU-bound: about 4.6 ms graph run for 16 frames when the card is shared with a trainer
(9 ms+ when busier), CPU phase about 1 ms per decision. Wall (8 worlds x 60 min, GPU shared with other jobs):
`serve` 955 s / 828 s, `inproc` 737 s / 617 s (vision); rule-bot lineups run all worlds in parallel threads with no
barriers. A 30 min x 8 run took 141 s (5.7 ms/call), i.e. about 19 min for 240 min. The 6-minute target is not reachable
with this fp32 net and 16 frames per call on a shared GPU (180k sequential calls x >= 3 ms); it needs a cheaper net
(fp16, smaller input) or an idle GPU. `--shards K` gives each process its own GPU time slice (helps only when the card is busy).

## M4 status (2026-10-08/09): fixing deaths, then PPO (branch `worktree-agent-ae99736f13458a832`)

**Result in one line.** Phase A (BC with a danger-state store) took the endless leader share from 42.0 to 51.4 and cut
in-env deaths/h from 15.6 to 9.2. Phase B (PPO) took it to 78.8 on the training field, **but most of that is the
field being known**: on a held-out field the PPO nets are ahead of bc3 by about 12-15 points (about 43 vs 28.5, 2 seed
sets), and a live server with a human and other strong bots did not show the gain (the coordinator's 9600 test, p2:
#1 3% of the time). Treat 78.8 as an over-fit upper bound and the held-out column as the honest number.

### Phase A: fixes in BC (`train_bc.py`, continued from bc3)
- **Danger / death-precursor store** (`--danger-cap 30000 --danger-frac 0.15..0.25`): in the collector, a frame is
  flagged when the learner's mass falls below 70% of its value within the next 30 decisions (2.4 s; a death counts)
  and the mass was at least 40. The flagged frames (current + previous, as a normal stack) go to a FIFO store that holds
  about 12 min of wall time, against about 20 s for the ring (ring shrunk to 256 rows). Each batch draws 15-25% from
  it; labels are still the planner's. About 1.3% of the frames are flagged. DAgger: the student drives 85-90% of the
  slots from the start (the start net is already bc3).
- a1 = bc3 + 1.5 h (lr 2e-4, danger-frac 0.15); a2 = a1 + 1.5 h (lr 1.5e-4, danger-frac 0.25). Teacher agreement on
  the held-out set falls a little (0.77 -> 0.73); as in M2 this is a distribution shift, not worse play.

| net | in-env deaths/h (seeds 11, 12, 33 min) | in-env leader | top-3 |
|---|---|---|---|
| planner (M2) | 8.0 | 0.207 | 0.52 |
| bc3 | 15.6 | 0.230 | 0.49 |
| a1 | 11.1 | 0.220 | 0.52 |
| a2 | 9.2 | 0.222 | 0.54 |
| p1 (PPO) | 6.4 | 0.273 | 0.62 |
| p2 | 5.6 | 0.316 | 0.65 |
| p3 | 5.5 | 0.301 | 0.65 |
| p4 | 5.8 | 0.288 | 0.65 |

### Phase B: PPO (`train_ppo.py`, then `train_ppo2.py`)
- Policy = categorical over the 64x64 heat-map cells, logits = heat / 0.5; the executed target is the cell centre.
  Split and eject are not used (always off). Reward per decision: d log(mass) (clipped to +-1) + 0.01 while #1 in the
  world - 3.0 on a death. GAE gamma 0.997, lambda 0.95, clip 0.1, 2 epochs, minibatch 512, lr 4e-5 (value head
  3e-4) with a cosine decay, KL early stop at 0.03, entropy 5e-4. 64 worlds x 4 learners, rollout 48 steps, bf16.
- Value head: pretrained for 30 iterations with the trunk frozen and the policy fixed (p1 only), then trained jointly.
- Anchor: `train_ppo.py` = cross-entropy to the planner's Gaussian heat label with a coefficient decaying 0.1 -> 0.02
  (p1) / 0.05 -> 0.01 (p2-p4); `train_ppo2.py` = KL(ref || pi) to a frozen BC net (a2), coefficient 0.2 -> 0.05, no
  planner labels.
- Chain: a2 -> p1 (2 h, 1077 it) -> p2 (2 h) -> p3 (2 h) -> p4 (2 h, plateau; no endless run) on the fixed
  endless field. In-env and training metrics rise up to p2-p3 and then flatten.
- No rollback was needed: each stage improved in-env. p4 did not improve (0.288 in-env), so it was not evaluated further.

### Endless leader share, training field (champ, r2g9, safe, ramp_r2, ramp300 + 8 server bots; 8 worlds x 240 min)

| spec | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|
| planner `plan_c34` | 46 | 35 | 38 | 38 | 44 | 40.2 |
| vision_bc3 (M3) | 48 | 36 | 40 | 42 | 44 | 42.0 |
| vision_a1 | 50 | 47 | 59 | 51 | 47 | 50.8 |
| vision_a2 | 59 | 45 | 51 | 47 | 55 | 51.4 |
| vision_p1 | 68 | 65 | 62 | 73 | 61 | 65.8 |
| vision_p2 | 78 | 70 | 64 | 65 | 71 | 69.6 |
| vision_p3 | 81 | 80 | 78 | 78 | 77 | **78.8** |

Each step beats the previous one on at least 4 of 5 sets (a2 vs a1 is a tie: 51.4 vs 50.8).

### Held-out field (not seen in training)
Lineup: `vis` x2, `plan` (plan_c34) x2, `plan2` (plan2_c34) x1, `bc3` (vision_bc3) x2, `apos` x1, `hvirus`
(hunter_virus) x1 + 8 server bots (`endlessH.sh`). The league pool in the p6/p7 runs is disjoint from it, except for the
frozen bc3 slots in `train_ppo2.py` (requested by the coordinator; bc3 stays in the held-out field, so p6/p7 are not
fully held out against it).

| spec | 3000 | 11000 | mean of 2 | leader share of the 2 bc3 copies in the same runs |
|---|---|---|---|---|
| vision_bc3 as candidate | 26 | 31 | 28.5 | (itself 22 / 24, plan 34 / 26) |
| vision_p2 | 38 | 47 | 42.5 | 47 / 44 |
| vision_p3 | 39 | 47 | 43.0 | 52 / 41 |
| vision_p6 (league PPO, frozen self-play) | 45 | 39 | 42.0 | 38 / 37 |
| vision_p7 (p6 + 2 h) | 36 | 45 | 40.5 | 48 / 43 |

Reading: against a field with vision opponents every PPO net gets about 40-45%, against 28.5% for bc3 (+8 to +20 per
seed). Head to head, the p2/p3 candidates are about even with their bc3 opponents (bc3 copies lead on seed 3000);
p6 is ahead of them on both seeds' family tables. The differences between p2, p3, p6 and p7 are inside the noise
(2 seeds, about +-5). Only 2 of 5 seed sets were run on the held-out field (a run takes 20-40 min with vision
opponents); the 5-set rule was not met for it.

### League PPO (`--league`, `train_ppo2.py`) and the Rust options
- `ogar_py.VecEnv(..., pool=[(name, spec, count)], pool_pick=k, giant_p=p, giant_min, giant_max)`: every world draws k
  entries of the pool (deterministic in the world seed); with probability p one opponent is a giant, fed (largest cell,
  up to 500 per 125 ticks) up to a target mass in [3000, 10000] whenever it falls below 70% of it. With a giant the
  learner is leader almost never (0.3% vs 12% in a test), so the leader bonus does not apply there.
  `rust/ogar/src/vision.rs`.
- Pool: the five endless-field bots, c3_4, shoot, hunter_hunt, hunter_reckless, five `runs/ab` c34 variants.
- Frozen self-play: 2 of 5 learner slots per world are driven by frozen nets (bc3, a2, plus 3-4 snapshots of the
  policy taken every 40-60 iterations, greedy decode). Vision opponents inside the env's in-process agents are not
  available (`ogar_py` is built without the `vision-net` feature), so this runs in the trainer on the GPU.
- Throughput of `train_ppo2.py` alone: rollout 5.7k decisions/s over all 320 slots (planner labels off), training
  3.0k samples/s (incl. the ref forward), 8.8 s per 9.2k-sample iteration; about 17 s while an endless eval ran at the
  same time. `train_ppo.py` (planner labels on) took 6 s per 12.3k samples. torch.compile was not tried.

### What limited it
1. **The training field.** Rule-bot opponents are fixed and exploitable; the policy specialises on them (78.8 vs about
   43 on the held-out field). Self-play / league came late and its benefit (p6 45/39, p7 36/45) cannot be separated
   from noise with 2 seeds.
2. **Eval cost.** One endless run takes 22 min (rule field) to 35 min (vision opponents) and shares the machine with training.
3. **Splitting stayed off** (the split head is poor and PPO never used it). No virus, split-kill or pop skill was
   trained or measured; the per-skill stats asked for (kills via split, pops caused, pieces eaten after a pop) were not
   collected, and the reward has no term for them.
4. Time: about 14 h of training wall time in total (budget 12 h); p4 and p7 showed plateaus.

### Files
`bot/vision_{a1,a2,p1,p2,p3,p6,p7}.json` (specs, `split` false), models in `runs/vision/export/vision_*.onnx`
(gitignored), checkpoints `runs/vision/{a1,a2,p1..p7}/` (gitignored), logs `logs/vision/*.jsonl`, eval JSONs
`runs/vision/evals/`. Scripts: `rl/vision/train_bc.py` (danger store), `train_ppo.py`, `train_ppo2.py`, `ppo_sum.py`,
`inenv_sum.py`, `endless5.sh` (training field), `endlessH.sh` (held-out), `queueH*.sh`.
Recommended: `vision_p3` (best on the training field, equal on the held-out field) or `vision_p6` for the
league-trained net.
