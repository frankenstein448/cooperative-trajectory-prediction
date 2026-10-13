#!/bin/bash
# runs/DECISION_RULES_TURNING_TEST.md: confirmatory test of "the map lowers minADE6 on turning vehicles"
# on the TEST split. Existing D / Dnomap epoch-12 checkpoints, nothing retrained.
#   1. eval_scenes.py --split test --confirmatory-turning-test, once per checkpoint -> eval_<run>_test_cond.log
#      (logged in runs/TEST_EVALS.log with purpose "confirmatory condition evaluation (turning hypothesis
#      from tuning)"; eval_scenes.py refuses a second one for the same checkpoint)
#   2. right after each run, its accuracy rows must match the original eval_<run>_test.log exactly;
#      any difference stops the script
#   3. analyze_conditions.py --rule turning_test -> analyze_turning_test.log
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
RUNS="D_seed0 D_seed1 D_seed2 Dnomap_seed0 Dnomap_seed1 Dnomap_seed2"
ROWS='^(Town0|ALL |CV |Sanity|Median|Scenes|    )'

for t in town03 town04 town05; do
  [ -f junction_waypoints_$t.npy ] || { echo "missing junction_waypoints_$t.npy (export_junctions.py)"; exit 1; }
done
for n in $RUNS; do
  [ -f eval_${n}_test.log ] || { echo "missing original eval_${n}_test.log"; exit 1; }
  [ -f runs/$n/checkpoint_epoch12.pt ] || { echo "missing runs/$n/checkpoint_epoch12.pt"; exit 1; }
  [ ! -e eval_${n}_test_cond.log ] || { echo "eval_${n}_test_cond.log already exists; refusing to overwrite"; exit 1; }
done

same_rows () {  # $1 = original log, $2 = re-run log
  if ! diff <(grep -E "$ROWS" "$1") <(grep -E "$ROWS" "$2"); then
    echo "MISMATCH: $2 accuracy rows differ from $1 -- stopping"; exit 1
  fi
  echo "accuracy rows identical: $2 vs $1"
}

for n in $RUNS; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt \
    --split test --confirm-final-test --confirmatory-turning-test 2>&1 | tee eval_${n}_test_cond.log
  same_rows eval_${n}_test.log eval_${n}_test_cond.log
done

python analyze_conditions.py --rule turning_test 2>&1 | tee analyze_turning_test.log
echo "TURNING TEST DONE"
