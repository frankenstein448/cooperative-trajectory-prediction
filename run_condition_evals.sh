#!/bin/bash
# runs/DECISION_RULES_CONDITIONS.md: condition-based evaluation (turning / straight / stationary /
# near_junction) of the existing epoch-12 checkpoints of D, Dnomap and EgoOnly. Evaluation only,
# tuning split only, nothing retrained.
#   1. eval_scenes.py per run -> eval_<run>_e12_cond.log (refuses to overwrite an existing log)
#   2. right after each run, its accuracy rows must match the original eval_<run>_e12.log exactly
#      (same checkpoint, same data => same numbers); any difference stops the script
#   3. analyze_conditions.py -> analyze_conditions.log
# Needs junction_waypoints_town0{3,4,5}.npy (export_junctions.py, committed).
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
RUNS="D_seed0 D_seed1 D_seed2 Dnomap_seed0 Dnomap_seed1 Dnomap_seed2 EgoOnly_seed0 EgoOnly_seed1 EgoOnly_seed2"
ROWS='^(Town0|ALL |CV |Sanity|Median|Scenes|    )'

for t in town03 town04 town05; do
  [ -f junction_waypoints_$t.npy ] || { echo "missing junction_waypoints_$t.npy (export_junctions.py)"; exit 1; }
done
for n in $RUNS; do
  [ -f eval_${n}_e12.log ] || { echo "missing original eval_${n}_e12.log"; exit 1; }
  [ -f runs/$n/checkpoint_epoch12.pt ] || { echo "missing runs/$n/checkpoint_epoch12.pt"; exit 1; }
  [ ! -e eval_${n}_e12_cond.log ] || { echo "eval_${n}_e12_cond.log already exists; refusing to overwrite"; exit 1; }
done

same_rows () {  # $1 = original log, $2 = re-run log
  if ! diff <(grep -E "$ROWS" "$1") <(grep -E "$ROWS" "$2"); then
    echo "MISMATCH: $2 accuracy rows differ from $1 -- stopping"; exit 1
  fi
  echo "accuracy rows identical: $2 vs $1"
}

for n in $RUNS; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt 2>&1 | tee eval_${n}_e12_cond.log
  same_rows eval_${n}_e12.log eval_${n}_e12_cond.log
done

python analyze_conditions.py 2>&1 | tee analyze_conditions.log
echo "CONDITION EVALUATION DONE"
