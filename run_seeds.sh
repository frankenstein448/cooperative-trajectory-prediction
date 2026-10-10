#!/bin/bash
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy"
run () {  # $1 = out-dir name, $2.. = extra flags
  name=$1; shift
  python -u train.py $FLAGS "$@" --out-dir runs/$name 2>&1 | tee train_$name.log
  python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
}
run D_seed1 --seed 1
run D_seed2 --seed 2
run Dnomap_seed0 --seed 0 --no-map
run Dnomap_seed1 --seed 1 --no-map
run Dnomap_seed2 --seed 2 --no-map
echo "ALL SEED RUNS DONE"
