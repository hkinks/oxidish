#!/bin/bash
# usage: scripts/export.sh run ckpt-tag  -> bots/models/vision_<run>_<tag>.onnx + bots/specs/vision_<run>_<tag>.json (with the hunter reflex)
# needs onnx + onnxscript next to torch
r=$1; t=$2; n=${r}_${t}
(cd training/rl && ${PYTHON:-python} export.py ../../runs/vision/$r/$t.pt ../../bots/models/vision_$n > /dev/null) || exit 1
printf '{"kind":"vision","model":"bots/models/vision_%s.onnx","split":false,"reflex_hunter":true,"reflex_hunter_genes":"bots/specs/best/rules.json"}\n' $n > bots/specs/vision_$n.json
ls -la bots/models/vision_$n.onnx
