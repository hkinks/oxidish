# Showcase server (real MultiOgar, port 9500)

A persistent, real-time arena for playing against the bots in the browser. It is shut down while
training runs in the simulator. Bring it back with:

```bash
uv run python -m arena.showcase --port 9500 --follow nn --food-mult 2 --log-players 5 >> logs/showcase.txt 2>&1
cd vendor/agarian-client && python -m http.server 8090 --bind 127.0.0.1          # game client: http://127.0.0.1:8090
uv run python -m arena.join --port 9500 --agents champ=runs/final_champion.json:2 r2g9=runs/r2/champion_009.json:2 \
    nn36=runs/rl/nn36.pt:1 g4net=runs/rl/g4_show.pt:2 route:1 field:1 nnbest=runs/rl/final_nn.pt:2
PYTHONIOENCODING=utf-8 uv run python -m arena.apos --port 9500 --count 2       # AposBots
uv run python -m arena.autojoin --port 9500 --interval 1200 >> logs/autojoin.txt 2>&1
```

- **`arena.autojoin`:** every 20 minutes it joins the newest `ckpt_*.pt` of each run listed in `runs/rl/autojoin.txt`, as detached processes named `<run>-<it>` (2 copies). Their PIDs are kept in `runs/rl/autojoin.json`. Joined bots never reconnect: after a server restart, restart the joins too.
- **Sim-trained nets** (`rl/ppo_sim.py`) play on the real server through `rl/nohint.NoHintEncoder`. `rl/policy.py` picks it for checkpoints in the sim layout. Nets with the two sim-only channels (`killable`, `density`) need those channels added to `rl/guided.py` first.
- **Stats:** `uv run python -m arena.showcase_stats` gives average and peak mass, deaths per hour and "avg vs you" per player from the `[players]` log lines. The human benchmark (2 h, 2026-10-04) was average ~7.3k and peak 20k. That log is kept as `logs/showcase-benchmark-2026-10-04.txt`.
- **Web-client extras** (`vendor/agarian-client`, see client-hook.patch):
  - `+` (or numpad `+`) adds 500 mass to your biggest cell (server opcode 0xCD in `arena/server/arena.js`). The server flags that life in its `[players]` log (`"b": 1`, not shown in game), and `arena.showcase_stats` leaves it out of the human benchmark.
  - Spectating: Q switches between following the largest player and the whole map (arena.js zooms free roam out to fit the map, centred).

## Native game client (Rust, macroquad)
Faster alternative to the browser client; same wire protocol (MultiOgar protocol 6), works with the live showcase and with any server speaking it.
```bash
cd rust/game_client && cargo build --release -j 2
./target/release/game_client --name YourNick     # default server ws://127.0.0.1:9600 (Rust ogar); --server ws://127.0.0.1:9500 for the Node showcase; --novsync uncaps FPS
```
- Start screen: type a nickname, Enter or click to play, Tab to spectate (follows the largest player; Q switches to the whole map and back). After death: Enter/click respawns, Tab spectates.
- Mouse steers, Space split, W eject (hold repeats), `+` / numpad `+` adds 500 mass (opcode 0xCD), wheel zoom, F11 fullscreen, Esc quits.
- Rendering interpolates cells over 120 ms like the web client; leaderboard top-right, score bottom-left, FPS top-left.
- Testing without eyes: `--selftest` (scripted bot: steers, eats, dies, respawns, +500, split, eject; logs `[auto]` lines), `--screenshot out.png [--shot-at SECS]`.

## Rust server (`ogar`, rust/ogar): faster showcase + headless evaluation
A Rust port of the MultiOgar FFA game (same rules, config.ini values, server bots, protocol 6, and the arena.js extras 0xC8/0xC9 lockstep agents, 0xCA follow, 0xCB spawn size, 0xCC food, 0xCD +500 mass). Statistically, not bit-for-bit, equivalent to Node (see docs/ogar.md). Rule bots can run **inside** the server, without websockets.
```bash
cargo build --release -j 2 --manifest-path rust/ogar/Cargo.toml
# live server on a free port (never reuse 9500 while the Node showcase runs); 25 ticks/s, 8 server bots, in-process rule bots
rust/ogar/target/release/ogar serve --port 9600 --bots 8 --food-mult 2 --map 7000 --follow champ --log-players 5 --control-port 9601 \
    --agents champ=runs/final_champion.json:2 r2g9=runs/r2/champion_009.json:2 safe=bot/champ_safe.json:1 route:1
```
- Play with the web client (`setServer("127.0.0.1:9600")`, i.e. `?ip=127.0.0.1:9600` if enabled, or `arena.watch --port 9600`) or `game_client --server ws://127.0.0.1:9600`.
- External bots still work: `arena.join --port 9600 ...` and `arena_client run --port 9600 ...` (lockstep 0xC8/0xC9, `--sync 2`).
- **Add / remove in-process bots while it runs** (no restart): type commands on the server's stdin, or over the control port (127.0.0.1 only, one line per command, one reply line):
  `add ramp=bot/r2g9_ramp800.json:2` | `remove champ` (name prefix) | `list` | `food 1.5` | `quit`. E.g. `echo "add ramp=bot/r2g9_ramp800.json:2" | nc 127.0.0.1 9601` (control port) or PowerShell `TcpClient`.
  More control commands: `speed X` (tick rate as a multiple of real time: 0.5, 1, 2, 4, 10; `0` = unthrottled; maps to max_tps = 25 * X; `speed` alone returns the current value), `pause` / `resume`, and `status` (one line: tick, measured tps, speed, paused, bots/players/agents/clients, food target and multiplier, `agentlist=name:mass,...`). `list` also shows speed and paused. Lockstep agents (0xC8/0xC9) keep working under speed changes and pause (the server just waits for their acks).
  Optional auth for non-local use: `--control-token T`; then every control line must be prefixed `T ` (e.g. `T status`), or send `auth T` once per connection. Without the flag no token is needed. The control port binds 127.0.0.1; `--control-bind ADDR` changes that (use only together with a token).
- `--max-tps 0` runs ticks unthrottled; `--seed N` makes the world deterministic (0 = time).

Headless matches (no network at all, rows in the `arena.tournament --out` format):
```bash
rust/ogar/target/release/ogar match --agents ramp300=runs/r2g9_ramp300.json:1 champ=runs/final_champion.json:1 \
    --ticks 22500 --food-mult 2 --seed 1 --matches 48 --threads 2 --out rows.json
uv run python -m arena.tournament --engine ogar --agents ... --matches 48 --workers 2 --ticks 22500 --food-mult 2   # also arena.verdict / arena.long_search --engine ogar
```
- **Zoom:** auto zoom is on by default (playing: size-based like the web client; spectating the whole map (Q): map fitted to the window; spectating a player: server scale). The mouse wheel switches to manual zoom (can zoom out until the whole map fits the window); `Z` re-enables auto zoom.
- **Settings (Esc or F1):** server address, nickname, auto zoom, show mass/names/grid/FPS, vsync (applies on restart), FPS cap (with vsync off), interpolation smoothing, cell outline, dark/light background, Quit. Saved to `%APPDATA%\agario-client\settings.json` (override with `--config PATH`). `--server` / `--name` override the saved values.
- **Server control (F2):** talks to the server's control port from a background thread (the render loop never blocks). Fields: control address (default = game server host, port + 1; saved) and optional token (saved). Shows tick/tps/speed/paused/food/agent masses (refreshed every 2 s) or the error (e.g. "control port not reachable"). Actions: speed presets 0.5x / 1x / 2x / 4x / 10x / max, Pause/Resume, food density -/+, add bot (pick a preset or any `bot/*.json`, `runs/ab/*.json`; count 1-4; paths are relative to the server's working directory), remove selected / all bots. F2 or Esc closes. Test flag: `--control-open`.
- **Whole field while playing:** the server only sends your view box. The Rust server (`rust/ogar`) has an opt-in `ogar serve ... --full-view` that sends human players every node of the map (off by default); zoom out with the wheel to see it all.
- Test flags: `--spectate`, `--whole-map`, `--settings-open`, `--control-open`, `--zoom Z`, `--config PATH`.
