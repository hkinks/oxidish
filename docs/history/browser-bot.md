> Historical: describes code removed from the repo; see [README.md](README.md).

# Browser bot (first phase)

The bot plays agar.io on a **local** server, never on the official agar.io servers, which ban bots. It plays in a visible Chromium window. The architecture is the same as the paperclips bot: code handles the fast control loop and Jev makes the judgment calls.

## Setup (removed)

It ran against a local Node MultiOgar server (`vendor/MultiOgar`, port 8443) and the agarian web client, patched with
`client-hook.patch` to expose a read-only `window.__bot = {cells, border, camera, canvas}`. Both lived in the
git-ignored `vendor/`; the patches were removed with the code.

## How it works

- **State.** `bot/game.py` reads cells (id, x, y, size, kind) plus the camera from `window.__bot` every 50 ms.
- **Control.** The bot moves the **real mouse** with Playwright, so the client sends it to the server like any player's input. Split is a real Space key press. Respawning calls the client's `play(name)`, which is the same as pressing Play.
- **Steering** (`bot/steer.py`, 20 Hz, code only). A potential field: food pulls, prey pulls (the hunt target extra strongly), threats push inside their danger radius (wider if they're heavy enough to split onto us), viruses push when we'd pop on them, and map borders push back.
- **Jev** (`bot/strategist.py`). One `choice` picks the mode: farm / flee / hide (only offered when a virus is visible) / chase prey pN. When a split-kill is possible, one `noul` asks "is splitting safe now?". Jev is asked when the set of nearby threats or prey changes, at most once a second and at least every 5 s while players are visible, and never when the screen is empty. Each call takes 220–410 ms. Decisions are logged to `logs/decisions-*.jsonl`.

## What came next

The "screen-only, human-like bot" idea shelved here became the vision bot, with frames rendered inside the Rust
server instead of a detector on screenshots ([../vision-bot.md](../vision-bot.md)).
