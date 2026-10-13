# Playing against the bots

The Rust server (`ogar serve`) runs the game in real time with bots inside it; the native client (`oxidish`) connects
over the MultiOgar wire protocol (protocol 6). Any MultiOgar-protocol web client works too.

## Server

```bash
cargo build --release                      # target/release/ogar and target/release/oxidish
target/release/ogar serve --port 9600 --bots 12 --food-mult 2 --map 10000 --control-port 9601 --log-players 5 \
    --agents vision=bots/specs/best/vision.json:4 vision-old=bots/specs/best/vision-old.json:4 \
             planner=bots/specs/best/planner.json:4 rules=bots/specs/best/rules.json:4
```

- `--agents name=spec.json:N` starts N in-process bots from a spec; the name prefix is what the leaderboard shows
  (`vision-0`, `vision-1`, ...). Vision specs need ONNX Runtime at run time (`ORT_DYLIB_PATH` on Linux,
  `onnxruntime.dll` next to `ogar.exe` on Windows); they use CUDA 12 when it loads and the CPU otherwise.
- `--bots N` adds MultiOgar's own server bots, `--food-mult` scales the food target, `--map` sets the map size.
- `--follow PREFIX` makes spectators follow the largest bot with that name prefix; `--log-players N` writes a
  `[players]` line with the top N every few seconds (input for `training.eval.showcase_stats`).
- `--max-tps 0` runs ticks unthrottled; `--seed N` makes the world deterministic (0 = time).
- `--full-view` (off by default) sends human players every node of the map instead of their view box; zoom out with
  the wheel to see it all.
- The game port binds `0.0.0.0` (`--bind` changes it); the control port binds `127.0.0.1`.

### Control port and stdin

Type commands on the server's stdin, or send them to the control port (one line per command, one reply line):

| command | does |
|---|---|
| `add vision=bots/specs/best/vision.json:2` | add in-process bots (paths relative to the server's working directory) |
| `remove vision` | remove bots by name prefix |
| `list` | bots, speed, paused |
| `speed X` | tick rate as a multiple of real time (0.5, 1, 2, 4, 10; `0` = unthrottled); `speed` alone prints it |
| `pause` / `resume` | stop and continue the world |
| `food X` | food density multiplier |
| `status` | one line: tick, measured tps, speed, paused, bots / players / agents / clients, food, `agentlist=name:mass,...` |
| `quit` | stop the server |

Any TCP client works, e.g. `echo status | nc 127.0.0.1 9601` (BSD netcat closes after the reply; with OpenBSD netcat
add `-N`). Lockstep agents (external bots using 0xC8/0xC9, see [training.md](training.md)) keep working under speed
changes and pause: the server waits for their acks.

For non-local use, start with `--control-token T`: every control line must then be prefixed `T ` (e.g. `T status`),
or send `auth T` once per connection. `--control-bind ADDR` moves the control port off 127.0.0.1; use it only with a
token.

## Native client (`oxidish`)

```bash
target/release/oxidish --name you                    # default server ws://127.0.0.1:9600
scripts/client.sh --name you --server ws://HOST:PORT  # builds it first if needed; sets up X11/GL on NixOS
```

- Start screen: type a nickname, Enter or click to play, Tab to spectate. After death: Enter / click respawns, Tab
  spectates. Spectating: Q switches between following the largest player and the whole map, Tab cycles players.
- Mouse steers, Space splits, W ejects (hold repeats), `M` adds 500 mass to your biggest cell (opcode 0xCD; the
  server flags that life in its `[players]` log and the human benchmark leaves it out), `+` / `-` step the server
  speed through the control port, wheel zooms, Z returns to auto zoom, F11 fullscreen, Esc quits.
- **Zoom:** auto by default (playing: size-based like the web client; whole map: fitted to the window; following a
  player: server scale). The wheel switches to manual zoom.
- **Settings (Esc or F1):** server address, nickname, auto zoom, mass / names / grid / FPS, vsync (applies on
  restart), FPS cap, interpolation smoothing, cell outline, dark / light background. Saved to
  `~/.config/agario-client/settings.json` on Linux and `%APPDATA%\agario-client\settings.json` on Windows
  (`--config PATH` overrides). `--server` / `--name` override the saved values. `--font ubuntu|nunito|inter|quicksand`
  or a `.ttf` path changes the font.
- **Server control (F2):** talks to the control port from a background thread. Fields: control address (default:
  game server host, port + 1) and optional token. Shows tick / tps / speed / paused / food / bot masses (every 2 s).
  Actions: speed presets 0.5x .. 10x / max, pause / resume, food -/+, add bots (presets vision, vision-old, planner,
  rules, champ, c3_4, or any spec in `bots/specs/` and `bots/specs/ab/`; 1-4 at a time), remove selected / all bots.
- Rendering interpolates cells over 120 ms like the web client; `--novsync` uncaps FPS.
- Without eyes: `--selftest` runs a scripted player (steers, eats, dies, respawns, +500, split, eject; logs `[auto]`
  lines); `--screenshot out.png [--shot-at SECS]` saves a frame and exits (it also runs the scripted player). Other
  test flags: `--spectate`, `--whole-map`, `--settings-open`, `--control-open`, `--zoom Z`. `--help` lists them.

## Recording human play (`--record`)

Every time a human eats a bot is a weakness to fix. Start the server with `--record DIR` (off by default) and the
session is written to `DIR` as gzip JSONL, flushed every 5 s:

```bash
target/release/ogar serve --port 9600 ... --record logs/human [--record-all]
uv run python -m training.eval.human_report logs/human --out docs/human/<date>.md      # --session STAMP, --top N
uv run python -m training.eval.showcase_stats --log logs/serve.txt   # server stdout with --log-players: mass, deaths per player
```

- `events-<stamp>.jsonl.gz`: every player cell eaten by another player (tick, eater / victim, kind
  human / agent / bot, masses, positions, whether the victim burst on a virus within 3 s) and a `top` row per second.
- `scenes-<stamp>.jsonl.gz`: **loss scenes**. When an in-process bot loses ≥ 30% of its mass within ~0.3 s, or dies,
  its last ~3 s of decisions are dumped (cells, nearby players and viruses, aim, split, plan mode) with the eat events
  and the spec. Only when a human was the eater, unless `--record-all`.
- `humans-<stamp>.jsonl.gz`: the humans' inputs and view per decision step (mouse target, split / eject, own cells,
  visible nodes).
- The report gives the human's leader share and mass stats (also without +500 lives), deaths, mass taken per bot
  family, the loss scenes classified (popped / split_kill / pieces / unseen / fled / no_flee, tagged wall / corner /
  human) and the worst scenes as text plus an ASCII map. Examples: [human/](human/).
- Code: `server/src/record.rs` (recorder), `training/eval/human_log.py` (reading, classification, rendering).
