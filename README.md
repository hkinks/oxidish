# oxidish

Bots that learn to play agar.io, and a Rust port of the MultiOgar server to train and fight them in.

![The live server spectated from the native client: the vision bots (m7 = vision, p6h = vision-old) hold most of the top ten](docs/img/spectate.png)

- **`ogar`**: a Rust port of the MultiOgar agar.io server (FFA). Speaks the MultiOgar protocol, so real agar.io-style
  web clients connect to it, and runs bots in-process with no websockets, which makes headless evaluation fast. As far
  as we know it is the only Rust port of Ogar/MultiOgar.
- **Bots** in three generations: hand-written rules with genes tuned by evolution, a lookahead planner, and **vision
  bots**: a CNN that sees grayscale frames rendered by the server and was trained with league PPO, plus a rule-based
  "hunter" reflex that eats freshly popped giants.
- **`oxidish`**: a native client (Rust, macroquad) for playing against them.

  ![Playing against the bots: the human (blue, split into four) is #5; the big cyan cell is planhv-2 (planner)](docs/img/play.png)
- Drills (eat a giant that just popped on a virus, escape a hunting human) and the training code.

## Results

The metric is **leader share**: how often a bot is #1 on the leaderboard, sampled every 15 s over 8 worlds × 4 game
hours per seed set, against opponents it never trained against (details in [docs/status.md](docs/status.md)).

| bot | what | leader share |
|---|---|---|
| `bots/specs/plan_c34.json` | lookahead planner | 40% |
| `bots/specs/best/planner.json` | planner + hunter | 50% |
| `bots/specs/vision_p6.json` | vision net, league PPO (`best/vision-old.json` adds the hunter) | 58% |
| **`bots/specs/best/vision.json`** | vision net, PPO with the hunter in the loop | **64%** |

The rule-bot rows are measured in a different field from the vision rows, so compare within each group. In an 11-hour
live session against a human, the best bot (vision, then called m7) was #1 for 30% of the time, more than anyone else. The human was #1
for 63% of the time they were alive, so a good human still wins.

The goal was one bot that is #1 80% of the time and can split-kill, virus-pop and eat exploded enemies. That was
**not** reached. Eating exploded enemies works; the other skills cost more than they gained. What worked, what didn't
and why is in [docs/learnings.md](docs/learnings.md), probably the most useful page here.

## Quick start

Prebuilt Windows and Linux builds (server, client, models, ONNX Runtime) are on the [releases page](../../releases).
To build from source you need Rust (stable); the vision bots also need ONNX Runtime 1.24.

```bash
# build the server and the client (one Cargo workspace, binaries in target/release/)
cargo build --release -p ogar -p oxidish

# server with the rule bots (no ONNX Runtime needed)
target/release/ogar serve --port 9600 --bots 8 --food-mult 2 --map 7000 --control-port 9601 \
    --agents planner=bots/specs/best/planner.json:4 rules=bots/specs/best/rules.json:4

# play
target/release/oxidish --name you --server ws://127.0.0.1:9600
```

With the vision bots (the models are in `bots/models/`; on Linux point `ORT_DYLIB_PATH` at
`libonnxruntime.so`, on Windows put `onnxruntime.dll` next to `ogar.exe`; the server is built with vision-net support by default). The nets use CUDA 12 when it loads and the
CPU otherwise; on the CPU, 20 bots still run at the full 25 ticks/s on a 16-thread machine.

```bash
OGAR_VISION_THREADS=4 target/release/ogar serve --port 9600 --bots 12 --food-mult 2 \
    --map 10000 --control-port 9601 --agents vision=bots/specs/best/vision.json:6 vision-old=bots/specs/best/vision-old.json:4 \
    planner=bots/specs/best/planner.json:6 rules=bots/specs/best/rules.json:4
```

- In the client: the mouse steers, Space splits, W ejects, Tab spectates, F2 opens server control (speed, pause, bots).
  `--font ubuntu|nunito|inter|quicksand` changes the font.
- While the server runs, `add` / `remove` / `list` / `speed` / `status` on its stdin or control port change bots and
  speed live ([docs/showcase.md](docs/showcase.md)).
- `scripts/client.sh [args]` builds and starts the client in one step; on NixOS it also supplies the X11/GL
  libraries macroquad loads at run time.

**Writing, testing and training your own bot:** [docs/training.md](docs/training.md) covers the three ways a bot can
connect (in-process spec, external websocket bot in any language, Python RL environment), evaluation, and training
far faster than real time.

## Layout

```
server/          ogar: the MultiOgar server port, headless matches, drills, frame renderer, vision-net inference
client/          oxidish: the native game client
bots/brain/      bot logic in Rust: rule bots, planner, hunter, reflexes (also an external websocket bot client)
bots/specs/      bot specs (JSON): genes, planner settings, vision model + reflexes; best/ holds the four champions
bots/models/     ONNX vision models used by the specs
training/env/    oxidish_env: Python bindings (PyO3) with vectorised game worlds for RL training
training/rl/     vision-net training: imitation, league PPO, ONNX export
training/eval/   leader-share evaluation, gene search, drill and session reports (python -m training.eval.<tool>)
scripts/         client.sh, train.sh, eval.sh, export.sh
docs/            design notes and results; start with docs/status.md
```

Earlier phases (a Playwright browser bot, a GPU simulator in PyTorch, PPO against the Node MultiOgar server) were
removed from the tree; the last commit that has them is tagged `python-era`.

The project was developed on Windows with an NVIDIA GPU, so some docs still show Windows paths.

## Licence

[WTFPL](LICENSE): do what you want. `server/` is a port of
[MultiOgar](https://github.com/agarian-2/MultiOgar), which is based on [Ogar](https://github.com/OgarProject/Ogar),
so it stays under Apache 2.0 ([LICENSE](server/LICENSE), [NOTICE](server/NOTICE)). The bundled fonts keep their
own licences. Not affiliated with agar.io or Miniclip; the bots are for local servers only.
