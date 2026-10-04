"""Spectator window on an arena server (needs the web client on :8090).

A panel (bottom-left) lets you pick whom the camera follows: click a
leaderboard name, type a name prefix (e.g. "champ", "g7"), or "auto" for the
largest player. "Play" joins the match yourself (mouse steers, Space splits,
W ejects); use a real-time match (--max-tps 25) for normal game speed. After
dying, "Spectate" returns to watching.
"""

import argparse
import time

from playwright.sync_api import sync_playwright

_PANEL_JS = r"""
() => {
  if (document.getElementById('watchPanel')) return;
  const follow = name => {
    window.__bot.send([0xCA, ...new TextEncoder().encode(name)]);
    document.getElementById('wpFollowing').textContent = name || 'auto (largest)';
  };
  const panel = document.createElement('div');
  panel.id = 'watchPanel';
  panel.style.cssText = `position:fixed;left:8px;bottom:8px;z-index:10000;width:230px;
    background:rgba(20,20,28,.88);color:#e8e8f0;font:12px/1.5 monospace;padding:8px;border-radius:6px`;
  panel.innerHTML = `<b>follow:</b> <span id="wpFollowing">auto (largest)</span>
    <div id="wpList" style="margin:6px 0"></div>
    <input id="wpName" placeholder="name prefix" style="width:120px;color:#000">
    <button id="wpGo">follow</button> <button id="wpAuto">auto</button>
    <div style="margin-top:6px"><input id="wpNick" value="human" style="width:90px;color:#000">
    <button id="wpPlay">Play</button> <button id="wpSpec">Spectate</button></div>`;
  document.body.appendChild(panel);
  const style = document.createElement('style');
  style.textContent = `
    #watchPanel button { background:#3b82f6; color:#fff; border:1px solid #93c5fd;
      border-radius:4px; padding:2px 8px; font:bold 12px monospace; cursor:pointer; }
    #watchPanel button:hover { background:#2563eb; }
    #watchPanel input { background:#fff; color:#111; border:1px solid #888;
      border-radius:4px; padding:2px 4px; font:12px monospace; }
    #watchPanel #wpList div { color:#93c5fd; }
    #watchPanel #wpFollowing { color:#fde047; }`;
  document.head.appendChild(style);
  // Keep game hotkeys (W, Space...) from firing while typing in the panel.
  for (const id of ['wpName', 'wpNick'])
    document.getElementById(id).addEventListener('keydown', e => e.stopPropagation());
  document.getElementById('wpGo').onclick = () => follow(document.getElementById('wpName').value.trim());
  document.getElementById('wpAuto').onclick = () => follow('');
  document.getElementById('wpPlay').onclick = () => play(document.getElementById('wpNick').value || 'human');
  document.getElementById('wpSpec').onclick = () => spectate();
  setInterval(() => {
    const list = document.getElementById('wpList');
    const names = (window.__bot.leaderboard || []).map(i => i.name || '').filter(Boolean);
    list.innerHTML = '';
    names.forEach((n, i) => {
      const a = document.createElement('div');
      a.textContent = `${i + 1}. ${n}`;
      a.style.cssText = 'cursor:pointer;text-decoration:underline';
      a.onclick = () => follow(n);
      list.appendChild(a);
    });
  }, 1000);
}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--client", default="http://127.0.0.1:8090/")
    args = p.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, args=["--start-maximized"])
        page = browser.new_page(no_viewport=True)
        page.goto(args.client)
        page.wait_for_function("window.__bot && window.__bot.canvas")
        page.evaluate("addr => setServer(addr)", f"127.0.0.1:{args.port}")
        time.sleep(1.5)
        page.evaluate("spectate()")
        page.evaluate(_PANEL_JS)
        try:
            while not page.is_closed():
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
