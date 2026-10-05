"""Shared RL setup: opponent pool, env configs, checkpoint IO."""

import json
from pathlib import Path

import torch

from .model import Policy

ROOT = Path(__file__).resolve().parent.parent


def champion_pool() -> list[dict]:
    """All evolved champions found under runs/, plus the hand-written baselines."""
    pool = [{"farm": "route"}, {"farm": "field"}]
    for path in sorted(ROOT.glob("runs/[rq]*/champion_[0-9]*.json")):
        pool.append(json.loads(path.read_text()))
    return pool


def latest_champion() -> dict:
    paths = sorted(ROOT.glob("runs/[rq]*/champion_[0-9]*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(paths[-1].read_text())


def final_champion() -> dict:
    return json.loads((ROOT / "runs" / "final_champion.json").read_text())


def env_cfgs(workers: int, n_learn: int, n_opp: int, episode_ticks: int, seed: int,
             teacher: dict | None = None, bots: int = 8, map_size: int = 7000,
             guide: dict | None = None, extra_opponents: list[dict] | None = None,
             strong: bool = False, opp_every: int = 2, league: str | None = None,
             virus_reward: tuple[float, float, float] = (0.0, 0.0, 0.0),
             explore_reward: float = 0.0) -> list[dict]:
    pool = champion_pool() + (extra_opponents or [])
    if guide and strong:  # only the guide champion itself and its hunter: the bots to beat
        from arena.policies import hunter_spec
        pool = [guide, guide, hunter_spec(guide)]
    elif guide:  # human-like aggression: hunters make up about a quarter of the opponents
        from arena.policies import hunter_spec
        pool += [hunter_spec(guide)] * max(2, len(pool) // 3)
    return [dict(idx=i, n_learn=n_learn, opponents=pool, n_opp=n_opp, episode_ticks=episode_ticks,
                 bots=bots, map_size=map_size, sync=2, teacher=teacher, seed=seed, guide=guide,
                 opp_every=opp_every, league=league,
                 virus_reward=virus_reward, explore_reward=explore_reward)
            for i in range(workers)]


def save(model: Policy, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config, **meta}, path)


def load(path: str | Path, device: str = "cpu") -> Policy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if cfg.get("kind") == "entity":                          # transformer over object tokens (rl.entity_model)
        from .entity_model import EntityPolicy
        model = EntityPolicy(**cfg).to(device)
    else:
        model = Policy(**cfg).to(device)  # old checkpoints: plain defaults
    # strict=False: older checkpoints (e.g. bc.pt) predate the separate critic, which then
    # starts fresh; the policy weights must all be present.
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if any(not k.startswith("critic_") for k in missing):
        raise ValueError(f"checkpoint {path} lacks policy weights: {missing}")
    model.guide = ckpt.get("guide")  # champion spec for guided nets, else None
    model.skip = ckpt.get("skip", 1)  # sim-trained nets: decides every `skip` steps, holding its move
    return model
