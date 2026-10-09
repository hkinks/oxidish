#!/bin/bash
# M5 phase 3: PPO from p6 with the split action (train_ppo3.py), run from the repo root
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-a1e64e46d058e79af/logs/pymod
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo3.py --name s2 --init runs/vision/p6/final.pt --ref runs/vision/a2/final.pt --hours 0.5 --worlds 32 --threads 5 --split-warm 60 --split-pos-w 100 --split-bc 0 --no-split --kill-bonus 0 --pop-bonus 0 --extra-frozen runs/vision/bc3/final.pt > logs/vision/s2.out 2>&1
