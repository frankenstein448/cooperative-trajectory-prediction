#!/bin/bash
# Step 2 of runs/DECISION_RULES_DEGRADED.md: train full and no-map under fog
# (D's flags + --degrade fog), seeds 0-2, 12 epochs; evaluate each at epoch 12
# under fog on the tuning split; then apply the pre-registered rule.
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --degrade fog"
run () {  # $1 = out-dir name, $2.. = extra flags
  name=$1; shift
  python -u train.py $FLAGS "$@" --out-dir runs/$name 2>&1 | tee train_$name.log
  python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt --degrade fog
}
for s in 0 1 2; do
  run Ffog_seed$s --seed $s
  run Ffog_nomap_seed$s --seed $s --no-map
done
python analyze_seeds.py --suffix tuning_fog --full Ffog_seed0,Ffog_seed1,Ffog_seed2 \
  --nomap Ffog_nomap_seed0,Ffog_nomap_seed1,Ffog_nomap_seed2 2>&1 | tee analyze_seeds_Ffog_tuning_fog.log
echo "FOG TRAINING DONE"
