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
./target/release/game_client --server ws://127.0.0.1:9500 --name YourNick     # --novsync uncaps FPS (default: monitor refresh)
```
- Start screen: type a nickname, Enter or click to play, Tab to spectate (follows the largest player; Q switches to the whole map and back). After death: Enter/click respawns, Tab spectates.
- Mouse steers, Space split, W eject (hold repeats), `+` / numpad `+` adds 500 mass (opcode 0xCD), wheel zoom, F11 fullscreen, Esc quits.
- Rendering interpolates cells over 120 ms like the web client; leaderboard top-right, score bottom-left, FPS top-left.
- Testing without eyes: `--selftest` (scripted bot: steers, eats, dies, respawns, +500, split, eject; logs `[auto]` lines), `--screenshot out.png [--shot-at SECS]`.
