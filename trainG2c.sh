#!/bin/bash
# M5 phase 3, recurrent: PPO with the split action + GRU memory + self-play + goal-aligned reward (train_ppo4.py)
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-a1e64e46d058e79af/logs/pymod
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo4.py --name g2c --init runs/vision/g1c/final.pt --ref runs/vision/a2/final.pt --hours 1.5 --mb 256 --worlds 32 --threads 5 --split-warm 40 --split-pos-w 30 --split-bc 0 --no-split --kill-bonus 0.02 --pop-bonus 0.05 --extra-frozen runs/vision/bc3/final.pt > logs/vision/g2c.out 2>&1
