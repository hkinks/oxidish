#!/bin/bash
# usage: ev.sh ckpt tag
for s in 11 12; do .venv/Scripts/python.exe rl/vision/eval_env.py --mode student --ckpt $1 --no-split --steps 24750 --seed $s --log-every 0 --out logs/vision/inenv_$2.jsonl > /dev/null 2>&1; done
