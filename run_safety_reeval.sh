#!/bin/bash
# Safety metrics (miss rate, off-road; eval_scenes.py) for the existing D / Dnomap seed
# checkpoints, epoch 12 -- the same checkpoints as the logged results, nothing retrained.
#   1. tuning split -> eval_<run>_e12_safety.log
#   2. test split, ONLY as a metrics-only re-run of the already-logged test evaluations
#      (eval_scenes.py --metrics-only-reeval refuses a run_dir/ckpt with no earlier
#      entry in runs/TEST_EVALS.log, and records the re-run there as metrics-only)
#      -> eval_<run>_test_safety.log
# Before each split moves on, every re-run's accuracy rows must match its original log
# exactly (same checkpoint, same data => same numbers); any difference stops the
# script before the test split is touched.
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
RUNS="D_seed0 D_seed1 D_seed2 Dnomap_seed0 Dnomap_seed1 Dnomap_seed2"
ROWS='^(Town0|ALL |CV |Sanity|Median|Scenes|    )'

same_rows () {  # $1 = original log, $2 = re-run log
  if ! diff <(grep -E "$ROWS" "$1") <(grep -E "$ROWS" "$2"); then
    echo "MISMATCH: $2 accuracy rows differ from $1 -- stopping"; exit 1
  fi
  echo "accuracy rows identical: $2 vs $1"
}

for n in $RUNS; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt 2>&1 | tee eval_${n}_e12_safety.log
done
for n in $RUNS; do same_rows eval_${n}_e12.log eval_${n}_e12_safety.log; done

for n in $RUNS; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt \
    --split test --confirm-final-test --metrics-only-reeval 2>&1 | tee eval_${n}_test_safety.log
done
for n in $RUNS; do same_rows eval_${n}_test.log eval_${n}_test_safety.log; done
echo "SAFETY RE-EVALUATION DONE"
