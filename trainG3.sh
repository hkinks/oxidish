#!/bin/bash
# M5 recurrent PPO, GRU params at the trunk lr. usage: trainG3.sh name [extra args]
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-a1e64e46d058e79af/logs/pymod
n=$1; shift
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo4.py --name $n --init runs/vision/p6g/init.pt --ref runs/vision/a2/final.pt --worlds 24 --threads 3 --mb 256 --extra-frozen runs/vision/bc3/final.pt "$@" > logs/vision/$n.out 2>&1
