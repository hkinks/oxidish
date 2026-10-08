#!/bin/bash
# wait for the old-field p3 run, then held-out runs for p2, bc3, p3
until [ -f runs/vision/evals/vision_p3_e41000.json ]; do sleep 10; done
sleep 5
for n in vision_p2 vision_bc3 vision_p3; do ./endlessH.sh $n; done
