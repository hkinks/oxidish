#!/bin/bash
# M8 league PPO, continuation of m7a. usage: trainM8.sh name hours [extra args]
# m7a plays in the league as a permanent frozen opponent WITH the hunter (--hunt-frozen); snapshots of the learner are added as it goes.
export PYTHONPATH=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-aa7a0565d655ef1b1/logs/pymod
n=$1; h=$2; shift; shift
/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe rl/vision/train_ppo5.py --name $n --init runs/vision/m7a/final.pt --ref runs/vision/p6/final.pt --hunt-frozen runs/vision/m7a/final.pt --hours $h --worlds 60 --threads 6 --snap-every 60 --snap-keep 6 --vpre 20 --seed 8100 "$@" > logs/vision/$n.out 2>&1
