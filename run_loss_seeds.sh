#!/bin/bash
# runs/DECISION_RULES_LOSS.md: L_seed{0..4} (D + --loss laplace_mixture) vs D_seed{0..4}, 12 epochs,
# each evaluated at epoch 12 on the tuning split; then the pre-registered rule.
#   1. D_seed0..2 (existing checkpoints, nothing retrained): re-evaluated for the ranking metrics
#      -> eval_D_seed{0,1,2}_e12_rank.log; their accuracy rows must match eval_D_seed*_e12.log exactly,
#      otherwise the script stops before any training.
#   2. D_seed3, D_seed4: new runs with D's exact flags.
#   3. L_seed0..4.
#   4. analyze_seeds.py --rule loss.
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy"
ROWS='^(Town0|ALL |CV |Sanity|Median|Scenes|    )'

same_rows () {  # $1 = original log, $2 = re-run log
  if ! diff <(grep -E "$ROWS" "$1") <(grep -E "$ROWS" "$2"); then
    echo "MISMATCH: $2 accuracy rows differ from $1 -- stopping"; exit 1
  fi
  echo "accuracy rows identical: $2 vs $1"
}
run () {  # $1 = out-dir name, $2.. = extra flags
  name=$1; shift
  python -u train.py $FLAGS "$@" --out-dir runs/$name 2>&1 | tee train_$name.log
  python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
}

for s in 0 1 2; do
  n=D_seed$s
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt 2>&1 | tee eval_${n}_e12_rank.log
  same_rows eval_${n}_e12.log eval_${n}_e12_rank.log
done

for s in 3 4; do run D_seed$s --seed $s; done
for s in 0 1 2 3 4; do run L_seed$s --seed $s --loss laplace_mixture; done

python analyze_seeds.py --rule loss --expect-n 2208 \
  --full L_seed0,L_seed1,L_seed2,L_seed3,L_seed4 \
  --nomap D_seed0:e12_rank,D_seed1:e12_rank,D_seed2:e12_rank,D_seed3,D_seed4 2>&1 | tee analyze_seeds_loss.log
echo "LOSS SEED RUNS DONE"
