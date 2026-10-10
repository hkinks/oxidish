#!/bin/bash
# usage: exportM8.sh run ckpt-tag  -> runs/vision/export/vision_<run>_<tag>.onnx + bot/vision_<run>_<tag>.json (with the hunter reflex)
# needs onnx + onnxscript: uv pip install --target logs/pyextra onnx onnxscript
R=/c/Users/hkink/Workspace/agario-bot/.claude/worktrees/agent-aa7a0565d655ef1b1
r=$1; t=$2; n=${r}_${t}
cd $R/rl/vision
PYTHONPATH=$R/logs/pyextra /c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe export.py ../../runs/vision/$r/$t.pt ../../runs/vision/export/vision_$n > /dev/null 2>&1
cd $R
printf '{"kind":"vision","model":"runs/vision/export/vision_%s.onnx","split":false,"reflex_hunter":true,"reflex_hunter_genes":"bot/drill/hv_commit_only.json"}\n' $n > bot/vision_$n.json
ls -la runs/vision/export/vision_$n.onnx
