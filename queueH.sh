#!/bin/bash
# old-field p3 seed 41000 (was interrupted), then held-out runs for p2, bc3, p3
[ -f runs/vision/evals/vision_p3_e41000.json ] || ./endless5.sh vision_p3 41000
for n in vision_p2 vision_bc3 vision_p3; do ./endlessH.sh $n; done
