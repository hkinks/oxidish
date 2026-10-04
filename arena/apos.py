"""Run Apostolique's AposBot (github.com/Apostolique/Agar.io-bot, MIT) on our server.

    uv run python -m arena.apos --port 9500 --count 2

AposBot is a strong open-source heuristic bot (flees, baits, lines up virus shots,
feeds viruses at enemies). Its userscript targets the official agar.io site, so we
load only its bot module (`createAposBot(api)`, module 9 of the bundle) into
headless pages of our patched client and give it the world API from BOT-API.md,
backed by `window.__bot`. Real time only: it plays at the speed of the page, so use
it on the showcase / realtime servers, not in lockstep training arenas.
"""

import argparse
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "vendor" / "apos" / "aposbot.user.js"

SHIM_JS = r"""
(name) => {
  const b = window.__bot;
  let pointX = 0, pointY = 0;
  const wrap = c => ({
    id: c.id, x: c.x, y: c.y, size: c.s, color: c.color || '#888888', name: c.name || '',
    isVirus: () => !!c.jagged,
    isNotMoving: () => c.nx === c.ox && c.ny === c.oy,
    getUptimeTime: () => c.updated || Date.now(),
  });
  const cellsById = () => {
    const out = {};
    for (const c of b.cells.byId.values()) if (!c.destroyed) out[c.id] = wrap(c);
    return out;
  };
  const api = {
    getPlayer: () => b.cells.mine.map(id => b.cells.byId.get(id)).filter(Boolean)
                      .map(wrap).sort((p, q) => q.size - p.size),
    getCells: cellsById,
    getCellsArray: () => Object.values(cellsById()),
    getMemoryCells: cellsById,
    getPointX: () => pointX, getPointY: () => pointY,
    setPoint: (x, y) => { pointX = x; pointY = y; },
    getMapStartX: () => b.border.left, getMapStartY: () => b.border.top,
    getMapEndX: () => b.border.right, getMapEndY: () => b.border.bottom,
    getX: () => b.camera.x, getY: () => b.camera.y,
    getRatio: () => b.camera.z, getZoomlessRatio: () => b.camera.z,
    getWidth: () => b.canvas.width, getHeight: () => b.canvas.height,
    getLastUpdate: () => Date.now(),
    getCurrentScore: () => 0,
    getMode: () => ':ffa',
    getServer: () => 'local',
    verticalDistance: () => Math.hypot(b.canvas.width, b.canvas.height) / b.camera.z,
    split: () => b.send([17]),
    shoot: () => b.send([21]),
    drawLine() {}, drawPoint() {}, drawCircle() {}, drawArc() {},
  };
  const bot = window.__aposRequire(9)(api);
  setInterval(() => {
    if (b.overlayShown || !b.cells.mine.length) { play(name); return; }
    try {
      const target = bot.mainLoop();
      if (target && target.length === 2) {
        pointX = target[0]; pointY = target[1];
        b.aim(pointX, pointY);
      }
    } catch (err) { console.error('[apos]', err); }
  }, 40);
}
"""


def bot_module_js() -> str:
    """The bundle without its site-hooking entry point, exposing its module loader."""
    src = BUNDLE.read_text(encoding="utf-8")
    src = re.sub(r"\n__require\(0\);\n\}\)\(\);\s*$", "\nwindow.__aposRequire = __require;\n})();\n", src)
    if "window.__aposRequire" not in src:
        raise RuntimeError("unexpected aposbot.user.js layout (entry call not found)")
    return src


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--name", default="apos")
    p.add_argument("--client", default="http://127.0.0.1:8090/")
    args = p.parse_args()
    module_js = bot_module_js()
    with sync_playwright() as pw:
        # Several bots share one browser: stop Chromium throttling timers in background tabs.
        browser = pw.chromium.launch(headless=True, args=[
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows"])
        pages = []
        for i in range(args.count):
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.on("console", lambda m: m.type == "error" and print("[page]", m.text, flush=True))
            page.goto(args.client)
            page.wait_for_function("window.__bot && window.__bot.canvas")
            page.evaluate("addr => setServer(addr)", f"127.0.0.1:{args.port}")
            page.add_script_tag(content=module_js)
            time.sleep(1.0)
            page.evaluate(SHIM_JS, f"{args.name}-{i}")
            pages.append(page)
        print(f"{args.count} AposBots playing on port {args.port}", flush=True)
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
