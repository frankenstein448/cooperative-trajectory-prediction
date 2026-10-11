#!/bin/bash
# Degraded-sensor evals (runs/DECISION_RULES_DEGRADED.md): the six clean-trained
# epoch-12 checkpoints, tuning split only, one pass per condition given (default: all four).
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
CONDITIONS="${*:-lidar_range cam_fog fog sensors_off}"
for x in $CONDITIONS; do
  for n in D_seed0 D_seed1 D_seed2 Dnomap_seed0 Dnomap_seed1 Dnomap_seed2; do
    python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt --degrade $x
  done
  python analyze_seeds.py --suffix tuning_$x 2>&1 | tee analyze_seeds_tuning_$x.log
done
echo "DEGRADED EVALS DONE"
