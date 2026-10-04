"""Play agar.io on the local MultiOgar server in a visible browser window.

Needs the game server (vendor/MultiOgar, port 8443) and the patched client
served on http://127.0.0.1:8090 — see docs/plan.md.

    uv run python -m bot.main [--tick 0.05] [--name jev-bot]
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from .farm import FoodRouter
from .game import Game
from .hud import Hud
from .steer import steer
from .strategist import Strategist
from .systemone import SystemOne

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="http://127.0.0.1:8090/")
    parser.add_argument("--server", default="127.0.0.1:8443")
    parser.add_argument("--name", default="jev-bot")
    parser.add_argument("--tick", type=float, default=0.05)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--base-url", default="https://api.typesafe.ai/v1")
    parser.add_argument("--farm", choices=("route", "field"), default="route",
                        help="pellet routing planner, or the original potential field")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        raise SystemExit("JEV_API_KEY missing (expected in .env)")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, args=["--start-maximized"])
        page = browser.new_page(no_viewport=True)
        page.goto(args.client)
        page.wait_for_function("window.__bot && window.__bot.canvas")
        page.evaluate("addr => setServer(addr)", args.server)
        time.sleep(1)

        game = Game(page, args.name)
        hud = Hud(page)
        strategist = Strategist(SystemOne(api_key, args.base_url, args.model),
                                hud.log, hud.show_decision, log_dir)
        router = FoodRouter() if args.farm == "route" else None
        lives, best, last_status, was_alive = 0, 0.0, 0.0, False
        try:
            while True:
                s = game.snapshot()
                if s is None or s.dead or s.main is None:
                    if was_alive:
                        hud.log(f"died (best mass {best:.0f})")
                        was_alive = False
                    strategist.reset()
                    if router:
                        router.reset()
                    game.respawn()
                    lives += 1
                    time.sleep(0.5)
                    continue
                was_alive = True
                best = max(best, s.total_mass)

                strategist.step(s)
                x, y, split = steer(s, strategist.plan, router)
                game.aim(s, x, y)
                if split:
                    time.sleep(0.08)  # let the client send the new mouse direction first
                    game.split()
                    hud.log("split attack!")
                hud.render(s.total_mass, best, lives, strategist.calls)

                if time.monotonic() - last_status > 15:
                    last_status = time.monotonic()
                    print(f"[status] mass={s.total_mass:.0f} cells={len(s.mine)} best={best:.0f} "
                          f"lives={lives} farm={args.farm} mode={strategist.plan.mode} jev_calls={strategist.calls}",
                          flush=True)
                time.sleep(args.tick)
        except KeyboardInterrupt:
            pass
        finally:
            browser.close()


if __name__ == "__main__":
    main()
