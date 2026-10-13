# agar.io Bot — Plan & Status

The bot plays agar.io on a **local** server, never on the official agar.io servers, which ban bots. It plays in a visible Chromium window. The architecture is the same as the paperclips bot: code handles the fast control loop and Jev makes the judgment calls.

## Setup

```
vendor/MultiOgar        game server (agarian-2/MultiOgar, Apache-2.0), port 8443
                        src/config.ini: serverPort = 8443, serverBots = 12
vendor/agarian-client   web client (agarian-2.github.io), patched with client-hook.patch
                        which adds a read-only window.__bot = {cells, border, camera, canvas}
```

`vendor/` is gitignored. To recreate it, clone both repos, apply `client-hook.patch` inside `agarian-client`, then run `npm install` in MultiOgar and apply the config edits above.

Run (three terminals):

```bash
cd vendor/MultiOgar/src && tail -f /dev/null | node index.js   # stdin must stay open, or the console crashes
cd vendor/agarian-client && python -m http.server 8090 --bind 127.0.0.1
uv run python -m bot.main
```

## How it works

- **State.** `bots/specs/game.py` reads cells (id, x, y, size, kind) plus the camera from `window.__bot` every 50 ms.
- **Control.** The bot moves the **real mouse** with Playwright, so the client sends it to the server like any player's input. Split is a real Space key press. Respawning calls the client's `play(name)`, which is the same as pressing Play.
- **Steering** (`bots/specs/steer.py`, 20 Hz, code only). A potential field: food pulls, prey pulls (the hunt target extra strongly), threats push inside their danger radius (wider if they're heavy enough to split onto us), viruses push when we'd pop on them, and map borders push back.
- **Jev** (`bots/specs/strategist.py`). One `choice` picks the mode: farm / flee / hide (only offered when a virus is visible) / chase prey pN. When a split-kill is possible, one `noul` asks "is splitting safe now?". Jev is asked when the set of nearby threats or prey changes, at most once a second and at least every 5 s while players are visible, and never when the screen is empty. Each call takes 220–410 ms. Decisions are logged to `logs/decisions-*.jsonl`.

## Next ideas

- Measure: survival time and peak mass per life. Compare against a no-Jev baseline (fixed rules) to see whether Jev adds anything.
- Kev-4B locally (~20 ms) would allow decisions at 5–10 Hz, and maybe a direction choice per tick.
- Merge awareness after splitting, using W (eject) to feed or bait, and teaming against the server bots.
- **On the shelf: a screen-only, human-like bot.** It would see only pixels, as a player does, instead of the client hook or server state.
  - **Detection:** a fine-tuned YOLO11/YOLOv8 nano (~1-3 ms per frame on the RTX 5090) finds cells, food, viruses and ejected mass. Plain OpenCV colour/circle detection is the cheap baseline.
  - **Training data:** labels come free by rendering frames from the simulator or the real server, which know every object's exact position and size.
  - **Pairing with Jev:** detections become JSON state for the trained net and for Jev's judgments (Jev takes text/JSON, not images; check the TypeSafe docs).
  - **Reusing the trained nets:** they read an object list through the encoder (`sim/obs.py`, `rl/guided.py`), so YOLO only has to produce that list. The rest has to be reconstructed:
    - "mine" from our own colour/name;
    - owner groups from colour/name;
    - velocities and merge timers from a tracker (ByteTrack), since they need object IDs across frames;
    - the camera from the grid lines and our own cells.

    Detections are noisy, so first train with noise in the sim (drop objects, jitter sizes, swap IDs), then do a short final round on real detector output.
  - **Where:** servers that allow bots only. The official agar.io servers ban them.
