# Testing and training bots

How a bot connects to the game, how to measure it, and how to train it faster than real time. Commands run from the
repo root. On Windows the binaries are `ogar.exe` / `arena_client.exe`.

```bash
cargo build --release --features vision-net --manifest-path rust/ogar/Cargo.toml   # server + evaluators (drop the feature without ONNX Runtime)
cargo build --release --manifest-path rust/arena_client/Cargo.toml                 # external Rust bot client
```

## Three ways to connect a bot

| | in-process agent | external websocket bot | `ogar_py` VecEnv |
|---|---|---|---|
| what | a spec JSON run inside `ogar` | your own program speaking the game protocol | Python RL environment, N worlds in one process |
| use for | rule bots, planner, exported vision nets; evaluation | bots in any language, quick experiments, Python policies | training nets (PPO, imitation) |
| speed | as fast as the engine (hundreds x real time) | lockstep, limited by your bot's latency (~75x real time measured) | ~45k learner decisions/s |
| real time needed | no | no (lockstep) | no |

### 1. In-process agents (spec files)

Every `ogar` command takes `--agents name=spec.json:N` (N copies, named `name-0` ... `name-N-1`), plus the built-ins
`route:N` and `field:N`. A spec is one of three kinds:

| kind | example | keys |
|---|---|---|
| rule bot | `bot/c3_4.json` | `"farm": "route" \| "field"`, `"params": {...}` (the genes, see `bot/params.py` / `rust/arena_client/src/params.rs`) |
| planner | `bot/plan_c34.json` | `"kind": "plan"`, `"base": <rule-bot spec used for the plan's rollouts>`, `"need_loss"` |
| vision net | `bot/vision_m7a_final.json` | `"kind": "vision"`, `"model": <.onnx>`, `"split"`, `"split_thr"`, `"eject"`, reflexes (below), `"ep"`, `"threads"` |

Vision reflexes are rule-based overrides on top of the net: `"reflex_hunter": true` with `"reflex_hunter_genes"`
(eat the pieces of a giant that just popped; the best bots use it), and `reflex_split`, `reflex_harvest`,
`reflex_virus`, `reflex_flee`, `reflex_guard` (each tested, each cost leader share; off in the best bots, see
[learnings.md](learnings.md)).

Vision nets need ONNX Runtime 1.24: `ORT_DYLIB_PATH=/path/libonnxruntime.so` (or `onnxruntime.dll` next to `ogar.exe`).
They use CUDA 12 when it loads and the CPU otherwise; `OGAR_VISION_EP=cpu|cuda|cuda_plain` forces one and
`OGAR_VISION_THREADS` sets the CPU threads per net (default 1; use 4-8).

### 2. External bots over websocket (any language)

`ogar serve` speaks MultiOgar protocol 6, the protocol agar.io-style web clients use, plus a lockstep extension so a
bot never misses a tick, however fast the server runs. All numbers are little-endian.

**Handshake (client to server):** `FE 06 00 00 00` (protocol 6), `FF 01 00 00 00` (key), then `C8` to register as a
lockstep agent. Without `C8` you are an ordinary real-time player.

**Each step:** every `--sync` ticks (default 2, i.e. 12.5 decisions per game second) the server sends every agent

    C9  u32 tick  u32 virus_shots  u32 virus_pops  f32 popped_mass      (17 bytes; the last three are cumulative stats)

and waits until every agent has answered with `C9 u32 tick` (or 2 s have passed). Before the ack, send the step's input:

| client to server | bytes | meaning |
|---|---|---|
| spawn | `00` name `00` | join / respawn (send when you have no cells) |
| mouse | `10 i32 x i32 y u32 0` | world point to move towards |
| split | `11` | Space |
| eject | `15` | W |
| spectate | `01` | |
| ack | `C9 u32 tick` | step done; the server may advance |
| follow | `CA` name-prefix | spectator camera follows a player whose name starts with it |
| spawn size | `CB u16 size` | cell size of this agent's next spawns (0 = default) |
| food | `CC u16 percent` | food density, 100 = the server's base |
| +500 mass | `CD` | humans only (not lockstep agents) |

| server to client | layout |
|---|---|
| `10` UpdateNodes | `u16 n_eat`, n x (`u32 killer`, `u32 victim`); records until `u32 id = 0`: `u32 id, i32 x, i32 y, u16 size, u8 flags`, then `u8 r,g,b` if flags & 0x02, skin `\0`-string if & 0x04, name `\0`-string if & 0x08; then `u16 n_remove`, n x `u32 id` |
| `20` AddNode | `u32 id`: this cell is yours |
| `12` ClearAll | forget all cells |
| `11` camera | `f32 x, f32 y, f32 scale` |
| `40` SetBorder | `f64 left, top, right, bottom` (+ game type and server name at the handshake) |
| `31` leaderboard, `63` chat, `FE` server stats | |

Cell flags: 0x01 virus, 0x20 ejected mass, 0x80 food, otherwise a player cell. Mass = size² / 100. The view is the
usual camera box around your cells (not the whole map).

Ready-made clients:

- **Python** `arena.client.StepConn`: the protocol above as `await next_step()`, `snapshot()`, `await act(x, y, split, eject)`.
  A complete bot:

  ```python
  import asyncio
  from arena.client import StepConn

  async def main():
      bot = StepConn("ws://127.0.0.1:9800", "mybot", sync=2)    # sync must match the server's --sync
      await bot.open()
      while True:
          await bot.next_step()
          s = bot.snapshot()                                     # s.mine, s.others: Cell(id, x, y, size, kind, name)
          if s.dead:
              await bot.act(0, 0)                                # respawns
              continue
          me = max(s.mine, key=lambda c: c.size)
          prey = [c for c in s.others if c.kind == "food" or (c.kind == "player" and c.size < me.size * 0.85)]
          t = min(prey, key=lambda c: (c.x - me.x) ** 2 + (c.y - me.y) ** 2, default=me)
          await bot.act(t.x, t.y)

  asyncio.run(main())
  ```

- **Rust** `arena_client run --port 9800 --agents planhv=bot/drill/plan_hv_commit_only.json:2 route:1 --ticks 6000 --sync 2`:
  plays rule-bot specs as external agents and prints avg / peak mass, deaths, kills.
- **Python** `python -m arena.join --port 9800 --agents name=spec.json:N`: rule bots that stay until you stop it.

### 3. `ogar_py` VecEnv (Python RL)

N worlds in one process, stepped in parallel on worker threads, with the GIL released. Frames are rendered in the
engine, so no websocket and no real time are involved.

```python
import numpy as np, ogar_py

env = ogar_py.VecEnv(8, 4,                                   # 8 worlds x 4 learners = env.n = 32
                     opponents=[("c34", "bot/c3_4.json", 2), ("planhv", "bot/drill/plan_hv_commit_only.json", 2)],
                     px=256, sync=2, bots=8, food_mult=2, map=7000, seed=1, threads=8, label=False)
obs = env.reset()                    # {"view": u8[N,2,256,256], "wide": u8[N,2,128,128]}  (this frame + the previous one)
a = np.zeros((env.n, 4), np.float32) # (u, v, split, eject): target in view-frame pixels 0..px; split / eject if > 0.5
obs, reward, done, info = env.step(a)
```

- `reward` = change of own mass (a death gives `-mass`); `done` = died and respawned (the obs is the new life's first
  frame). `info`: `mass`, `rank` (0 = largest player), `leader`, `ncells`, `teacher` (the teacher bot's action for this
  observation, for imitation learning), `hunter`, `events`.
- Opponents are in-process rule bots / planners (not vision nets; frozen nets play learner slots from Python instead,
  as `train_ppo5.py` does). `pool=[...]`, `pool_pick=k` draws k opponents per world from a league;
  `giant_p` adds a fed giant to that share of worlds; `chaser_p` adds a scripted human-like hunter.
- `teacher="bot/plan_c34.json"` with `label=True` computes teacher labels; `env.set_teacher(mask)` lets the teacher drive
  chosen learners (DAgger). `hunter="bot/drill/hv_commit_only.json"` with `env.set_hunter(mask)` lets the hunter
  reflex take over during its commit phase (flagged in `info["hunter"]`).
- Full reference: [vision-bot.md](vision-bot.md), M1 and M4-M7.

## a) Testing a bot

**Watch it.** Start a server with your bot and the best bots, then join with the client (or spectate with Tab):

```bash
rust/ogar/target/release/ogar serve --port 9800 --control-port 9801 --bots 8 --food-mult 2 --map 7000 \
    --agents mine=bot/my_bot.json:2 m7=bot/vision_m7a_final.json:2 planhv=bot/drill/plan_hv_commit_only.json:2
scripts/client.sh --server ws://127.0.0.1:9800
```

The server reads one command per line from its stdin or from the control port (127.0.0.1 only;
`--control-token T` adds a password). F2 in the client opens the same controls on the game port + 1, so use
`--control-port` = `--port` + 1.

| command | |
|---|---|
| `add name=spec.json:N` / `remove <name prefix>` / `list` | change bots without a restart |
| `speed X` | multiple of real time (0.5, 1, 4, 10, ...); `speed 0` = as fast as possible |
| `pause` / `resume` | |
| `status` | tick, measured tps, speed, players, `agentlist=name:mass,...` |
| `food <mult>`, `leaderboard`, `agents`, `quit`, `help` | |

```bash
python3 -c 'import socket,sys; s=socket.create_connection(("127.0.0.1",9801)); s.sendall((sys.argv[1]+"\n").encode()); print(s.makefile().readline())' status
```

`--record DIR --record-all` logs eat events and the agents' losses for `arena.human_report` / `rl/vision/lossdiag.py`.

**Head-to-head matches** (fresh worlds, fixed length, several matches in parallel):

```bash
rust/ogar/target/release/ogar match --agents mine=bot/my_bot.json:2 c34=bot/c3_4.json:2 route:1 \
    --ticks 22500 --food-mult 2 --seed 1 --matches 48 --threads 8 --out rows.json
```

prints log-mass, average and peak mass, deaths and kills per policy. Same seeds give the same matches, so variants can be
compared on identical games (`arena.tournament --engine ogar` adds confidence intervals).

**Leader share**, the project's main metric: the share of 15-s samples in which a bot family is #1 on the leaderboard,
in endless worlds that never reset (giants keep growing, dead players respawn):

```bash
python -m arena.endless --engine inproc --threads 8 --cpus 8 --candidate mine \
    --agents mine=bot/my_bot.json:2 p6h=bot/vision_p6_hunt.json:2 planhv=bot/drill/plan_hv_commit_only.json:2 \
             c34h=bot/drill/hv_commit_only.json:2 plan=bot/plan_c34.json:1 \
    --worlds 8 --game-min 240 --seed-base 3000 --out ls.json --stats-out ls.stats.json
```

- The published numbers are 8 worlds x 240 game-minutes per seed set, sets 3000, 11000, 21000, 31000, 41000. One set
  is noisy (about ±5 points), so a change counts only if the mean improves and it wins at least 3 of 5 sets.
- `scripts/eval.sh 2 <spec name> <seed>` runs exactly the "champions" field above; field 1 is the held-out field the
  vision results in [status.md](status.md) use.
- Rule bots run this in seconds to minutes. Vision nets need a GPU for full-size runs: on a CPU each batched inference
  call takes ~50 ms, so a full seed set takes many hours. For a quick CPU check, use fewer worlds and minutes.
- `ogar endless ...` (same options, `--out samples.json`) is the underlying evaluator without the Python summary.

**Drills**, for single skills with many paired repetitions:

```bash
# eat a giant that just popped on a virus
rust/ogar/target/release/ogar drill --agents mine=bot/my_bot.json c34h=bot/drill/hv_commit_only.json oracle=oracle \
    --scenario giant_popped --our-mass 500 --giant-mass 5000 --dist 600 --giant-mode bot --reps 100 --seed 1 --threads 8 --out drill.json
python -m arena.drill_report drill.json
# escape a human-like chaser
rust/ogar/target/release/ogar escape --agents mine=bot/my_bot.json c3_4=bot/c3_4.json --our-mass 1000,3000 --ratio 2,4 \
    --gap 1200 --place open,wall,corner --reps 30 --seed 1 --threads 8 --horizon 60 --out esc.json
python -m arena.escape_report esc.json
```

Details: [harvest-drill.md](harvest-drill.md), [escape-drill.md](escape-drill.md).

## b) Training faster than real time

| route | how it runs fast | measured here (16 threads, machine partly busy) |
|---|---|---|
| live server, in-process bots | `--max-tps 0` or `speed 0` | ~9,500 ticks/s (380x) with 2 rule bots + 8 server bots |
| live server, external lockstep bots | `speed 0`; lockstep waits for every agent's ack | 75x real time with 3 external Rust bots |
| `ogar match` / `ogar endless` | no clock at all, worlds on threads | 290x real time per thread (rule bots) |
| `ogar_py` VecEnv | no clock, no network, worlds on threads | 45k learner decisions/s = 114x real time per world (8 worlds, 32 learners, random actions) |

The server's tick rate is `--max-tps` (25 = real time; 0 = unthrottled) at startup, and `speed X` (`max_tps = 25 x X`)
while it runs. Lockstep agents keep working at any speed and while paused, because the server waits for their acks.
Human clients get at most ~40 updates/s, so you can watch a sped-up game.

### Rule bots: gene search

`arena.endless_search` runs CMA-ES over a rule bot's genes, scored by leader share in endless Rust-engine worlds:

```bash
python -m arena.endless_search --run es1 --init bot/c3_4.json \
    --field champ=runs/final_champion.json:2 planhv=bot/drill/plan_hv_commit_only.json:2 --popsize 8 --worlds 12 --game-min 120
```

Each generation is written to `runs/es1/gen_###.json` and the best genome to `runs/es1/best.json`, which is a spec
you can pass to `--agents`. `arena.evolve`, `arena.qd` and `arena.long_search` are the older searches on the Node
MultiOgar server; they need `vendor/MultiOgar` (see [plan.md](plan.md)).

### Vision bots: imitation, then league PPO

Training needs an NVIDIA GPU (the trainers keep their buffers on CUDA) and the Python setup below.

1. **Held-out set** for imitation: `python rl/vision/make_val.py --out runs/vision/val --windows 4 --worlds 64 --learners 4 --threads 10`
2. **Imitation + DAgger** of the planner: `python rl/vision/train_bc.py --name bc1 --hours 1.5`. The result is
   `runs/vision/bc1/final.pt`.
3. **League PPO with the hunter in the loop**, the recipe that made m7a:

   ```bash
   PYTHON=python scripts/train.sh m7 --init runs/vision/bc1/final.pt --ref runs/vision/bc1/final.pt \
       --extra-frozen runs/vision/bc1/final.pt --worlds 40 --hours 3 --vpre 20 --snap-every 40
   ```

   - m7a itself started from the p6 checkpoint (`scripts/train.sh` defaults to `runs/vision/p6/final.pt`), which is
     not in the repo. The arguments above start from your own imitation net instead.
   - Each world has 5 learner slots. 3 learn and 2 are played by frozen nets: the anchor, `--extra-frozen`, and a
     snapshot of the policy every `--snap-every` iterations. The opponents are drawn from a league of rule bots and
     planners (5 per world, a fed giant in 25% of worlds).
   - Reward: change of log mass, +0.05 while #1, a bonus for eating big players, a penalty for mass lost to players,
     and -3 per death. A KL anchor to `--ref` decays from 0.2 to 0.01.
   - While the hunter reflex is committed it drives the learner, and those decisions are masked out of the policy
     loss.
   - Checkpoints go to `runs/vision/m7/`, the log to `logs/vision/m7.jsonl`. All flags: `python rl/vision/train_ppo5.py -h`.

4. **Export and use** (needs `uv pip install onnx onnxscript`): `PYTHON=python scripts/export.sh m7 final` writes `runs/vision/export/vision_m7_final.onnx` and
   the spec `bot/vision_m7_final.json` (with the hunter reflex). That spec works in every command above (`--agents
   m7=bot/vision_m7_final.json:2`). Then measure it with `scripts/eval.sh` against p6_hunt and m7a.

What worked and what did not (reward terms, split training, recurrent nets, chasers): [learnings.md](learnings.md),
[vision-bot.md](vision-bot.md).

## Python setup

```bash
uv sync                                                            # torch comes from the CUDA 12.8 index (pyproject.toml)
uv run maturin develop --uv --release -m rust/ogar_py/Cargo.toml   # builds ogar_py into .venv
uv run python -m pytest rl/vision/test_env.py -q                   # VecEnv smoke tests
```

Then run the commands above with `uv run python ...`, or activate `.venv`. For the server-side tools only
(`arena.endless`, `arena.join`, `StepConn`, `endless_search`) a light venv with `websockets numpy cma` is enough; torch
is needed for training and export. On NixOS, pip wheels such as numpy need `libstdc++` and `zlib` on `LD_LIBRARY_PATH`
(for example from `nixpkgs#stdenv.cc.cc.lib` and `nixpkgs#zlib`).
