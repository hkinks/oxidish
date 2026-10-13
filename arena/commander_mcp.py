"""MCP server (stdio) that lets a Claude session command the bots of an ogar server over its control port.

Register it yourself (this repo does not touch any config):

    claude mcp add agar-commander -- uv run --no-project --with "mcp<2" python arena/commander_mcp.py --port 9601

Environment: OGAR_CONTROL_HOST / OGAR_CONTROL_PORT / OGAR_CONTROL_TOKEN override the defaults (127.0.0.1:9601, no token).
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

from arena.commander import Control  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=os.environ.get("OGAR_CONTROL_HOST", "127.0.0.1"))
ap.add_argument("--port", type=int, default=int(os.environ.get("OGAR_CONTROL_PORT", "9601")))
ap.add_argument("--token", default=os.environ.get("OGAR_CONTROL_TOKEN"))
args, _ = ap.parse_known_args()
ctl = Control(args.host, args.port, args.token)

mcp = FastMCP("agar-commander", instructions=(
    "Command the rule-bot / planner / vision agents of an agar.io ogar server. Call list_agents, then scene(agent) to see "
    "what an agent sees, then order(agent, verb, arg, ttl). Orders expire (default 10 s game time) and never override the "
    "bot's reflexes: it will not walk into a bigger cell."))

VERBS = "hunt <player name or #cell_id> | harvest <target> | virus-split <target> | goto <x> <y> | farm | flee | hide | split on|off | free"


@mcp.tool()
def list_agents() -> dict:
    """The server's agents: name, kind (rule/plan/vision), alive, mass, cells, active order; plus game time and speed."""
    return ctl.json("agents")


@mcp.tool()
def scene(agent: str) -> dict:
    """What `agent` sees: own mass/cells/position, top-10 leaderboard, visible players (relative x/y, gap, can_eat_us,
    we_can_eat, split_kill, split_threat, popped), viruses, wall distances and the active order. y points down."""
    return ctl.scene(agent)


@mcp.tool()
def leaderboard() -> list:
    """Top 10 players of the world by mass (names, mass, cells)."""
    return ctl.json("leaderboard")


@mcp.tool()
def order(agent: str, verb: str, arg: str = "", ttl: float = 10.0) -> str:
    """Give `agent` a standing order for `ttl` seconds of game time. verb: hunt | harvest | virus-split (arg = player
    name or #cell_id), goto (arg = "x y"), farm | flee | hide, split (arg = on|off), free (clear the order)."""
    line = f"order {agent} {verb} {arg}".rstrip() + ("" if verb == "free" else f" ttl={ttl:g}")
    return ctl.cmd(line)


@mcp.tool()
def orders() -> str:
    """The active orders of all agents."""
    return ctl.cmd("orders")


@mcp.tool()
def add_agent(spec: str, count: int = 1) -> str:
    """Add rule/plan/vision agents. spec = `name=path/to/genome.json` (server-side path) or `route`/`field`."""
    return ctl.cmd(f"add {spec}:{count}")


@mcp.tool()
def set_speed(x: float) -> str:
    """Game speed as a multiple of real time (1 = real time, 0 = unthrottled)."""
    return ctl.cmd(f"speed {x:g}")


@mcp.tool()
def pause() -> str:
    """Pause the game."""
    return ctl.cmd("pause")


@mcp.tool()
def resume() -> str:
    """Resume the game."""
    return ctl.cmd("resume")


@mcp.tool()
def frame(agent: str) -> Image:
    """The grayscale 256 px frame around `agent` (the image the vision bot sees)."""
    r = ctl.cmd(f"frame {agent}")
    if not r.startswith("png-base64:"):
        raise RuntimeError(r)
    return Image(data=base64.b64decode(r[len("png-base64:"):]), format="png")


if __name__ == "__main__":
    mcp.run()
