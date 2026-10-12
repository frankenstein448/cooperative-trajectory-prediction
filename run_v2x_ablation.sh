#!/bin/bash
# runs/DECISION_RULES_V2X.md: V2X ablation, reusable for the final model.
#   $1  base training flags (the full model's flags, without --seed / --out-dir / ablation flags)
#   $2  run-name prefix ("" for the D study; e.g. "F_" for the final model)
#   $3  full-model run name stem; the comparison runs are $3_seed0,$3_seed1,$3_seed2 (e.g. D)
#   $4  ALL sample count the logs must show (default 2208 = D's sample filter; 1984 with 8 history frames)
#   $5  log suffix of the full-model runs (default e12; e12_safety for D, whose original e12 logs
#       predate the safety metrics -- run_safety_reeval.sh writes those, with identical accuracy rows)
# Order: EgoOnly x3, NoCPM x3, NoCoop x3. After each config's three runs: eval_scenes.py on
# epoch 12 (tuning), then analyze_seeds.py --rule component against the full-model runs.
# D study:
#   ./run_v2x_ablation.sh "--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy" "" D 2208 e12_safety
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: $0 \"<base flags>\" <run prefix> <full-model run stem> [expect_n] [full log suffix]"; exit 2
fi
BASE_FLAGS=$1
PREFIX=$2
FULL_STEM=$3
EXPECT_N=${4:-2208}
FULL_SUFFIX=${5:-e12}
FULL="${FULL_STEM}_seed0,${FULL_STEM}_seed1,${FULL_STEM}_seed2"
for n in ${FULL//,/ }; do
  [ -f "eval_${n}_${FULL_SUFFIX}.log" ] || { echo "missing eval_${n}_${FULL_SUFFIX}.log for the full model"; exit 1; }
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

config () {  # $1 = config name, $2.. = ablation flags
  cfg=$1; shift
  for s in 0 1 2; do
    name=${PREFIX}${cfg}_seed$s
    python -u train.py $BASE_FLAGS "$@" --seed $s --out-dir runs/$name 2>&1 | tee train_$name.log
  done
  for s in 0 1 2; do
    name=${PREFIX}${cfg}_seed$s
    python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
  done
  python analyze_seeds.py --rule component --expect-n $EXPECT_N --full $FULL --full-suffix $FULL_SUFFIX \
    --nomap ${PREFIX}${cfg}_seed0,${PREFIX}${cfg}_seed1,${PREFIX}${cfg}_seed2 2>&1 | tee analyze_seeds_${PREFIX}${cfg}.log
}

config EgoOnly --no-cpm --no-coop
config NoCPM --no-cpm
config NoCoop --no-coop
echo "V2X ABLATION DONE"
