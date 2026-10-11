#!/bin/bash
# runs/DECISION_RULES_HISTORY.md: control Dh (D + require 8 history frames, no history
# input) vs H8 (D + 8-frame history, GRU encoder), seeds 0-2, 12 epochs; evaluate each
# at epoch 12 on the tuning split; then apply the pre-registered rule.
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy"
run () {  # $1 = out-dir name, $2.. = extra flags
  name=$1; shift
  python -u train.py $FLAGS "$@" --out-dir runs/$name 2>&1 | tee train_$name.log
  python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
}
for s in 0 1 2; do
  run Dh_seed$s --seed $s --require-history-frames 8
  run H8_seed$s --seed $s --history-frames 8
done
python analyze_seeds.py --rule history --expect-n 1984 --full H8_seed0,H8_seed1,H8_seed2 \
  --nomap Dh_seed0,Dh_seed1,Dh_seed2 2>&1 | tee analyze_seeds_history.log
echo "HISTORY SEED RUNS DONE"
