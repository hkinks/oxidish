"""On-page overlay showing the current Jev plan and recent events."""

import time

_RENDER_JS = """
([head, decision, log]) => {
  let hud = document.getElementById('botHud');
  if (!hud) {
    hud = document.createElement('div');
    hud.id = 'botHud';
    hud.style.cssText = `position:fixed;top:8px;left:50%;transform:translateX(-50%);
      width:380px;z-index:9999;pointer-events:none;background:rgba(20,20,28,.85);
      color:#e8e8f0;font:12px/1.4 monospace;padding:8px 10px;border-radius:6px;white-space:pre-wrap`;
    document.body.appendChild(hud);
  }
  hud.textContent = head + '\\n' + decision + '\\n' + log;
}
"""

MAX_LOG_LINES = 5


class Hud:
    def __init__(self, page):
        self._page = page
        self._lines: list[str] = []
        self._decision = "Jev: (no decision yet)"

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self._lines = (self._lines + [line])[-MAX_LOG_LINES:]

    def show_decision(self, label: str, probs: dict) -> None:
        bars = "\n".join(f"  {p:4.0%} {'█' * round(p * 20):<20} {name}"
                         for name, p in sorted(probs.items(), key=lambda kv: -kv[1]))
        self._decision = f"Jev → {label}" + (f"\n{bars}" if bars else "")
        self.log(f"Jev → {label}")

    def render(self, mass: float, best: float, lives: int, calls: int) -> None:
        head = f"🤖 jev-bot  mass {mass:.0f}  best {best:.0f}  life #{lives}  Jev calls {calls}"
        self._page.evaluate(_RENDER_JS, [head, self._decision, "\n".join(self._lines)])
