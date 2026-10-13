#!/bin/bash
# usage: scripts/export.sh run ckpt-tag  -> runs/vision/export/vision_<run>_<tag>.onnx + bot/vision_<run>_<tag>.json (with the hunter reflex)
# needs onnx + onnxscript next to torch
r=$1; t=$2; n=${r}_${t}
(cd rl/vision && ${PYTHON:-python} export.py ../../runs/vision/$r/$t.pt ../../runs/vision/export/vision_$n > /dev/null) || exit 1
printf '{"kind":"vision","model":"runs/vision/export/vision_%s.onnx","split":false,"reflex_hunter":true,"reflex_hunter_genes":"bot/drill/hv_commit_only.json"}\n' $n > bot/vision_$n.json
ls -la runs/vision/export/vision_$n.onnx
