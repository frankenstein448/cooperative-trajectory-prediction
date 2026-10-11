#!/bin/bash
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
for n in D_seed0 D_seed1 D_seed2 Dnomap_seed0 Dnomap_seed1 Dnomap_seed2; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt \
    --split test --confirm-final-test 2>&1 | tee eval_${n}_test.log
done
python analyze_seeds.py --suffix test 2>&1 | tee analyze_seeds_test.log
echo "TEST EVALS DONE"
