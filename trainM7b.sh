#!/bin/bash
# M7b: continue m7a with the scripted chaser in a fraction of the worlds. usage: trainM7b.sh name [extra args]
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-af25e36448856cf44/logs/pymod2
n=$1; shift
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo5.py --name $n --init runs/vision/m7a/final.pt --ref runs/vision/p6/final.pt --threads 4 "$@" > logs/vision/$n.out 2>&1
