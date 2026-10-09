#!/bin/bash
# M7 league PPO with the hunter reflex in the loop. usage: trainM7.sh name [extra args]
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-af25e36448856cf44/logs/pymod
n=$1; shift
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo5.py --name $n --init runs/vision/p6/final.pt --ref runs/vision/p6/final.pt --threads 4 "$@" > logs/vision/$n.out 2>&1
