#!/bin/bash
# runs/DECISION_RULES_CORRECTED.md: corrected Part 1 rerun on split v2 with map_transform mirror.
# Order: C_Dnomap x3, C_EgoOnly x3 (faster), C_D x3. After each config's three runs: eval_scenes.py on
# epoch 12 (tuning split v2) -> eval_<run>_e12.log. After C_D: the map rule, the V2X rule and the
# secondary turning hypothesis (analyze_*_corrected*.log).
#   ./run_corrected_seeds.sh
set -euo pipefail
BASE_FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --map-transform mirror --split-version v2"

# the setup must be committed: every run records git rev-parse HEAD
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json runs/DECISION_RULES_CORRECTED.md)" ]; then
  echo "uncommitted changes to tracked code / split / rules; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch test_scenes_v2.json runs/DECISION_RULES_CORRECTED.md > /dev/null
for cfg in Dnomap EgoOnly D; do
  for s in 0 1 2; do
    [ ! -e runs/C_${cfg}_seed$s ] || { echo "runs/C_${cfg}_seed$s exists; refusing to overwrite"; exit 1; }
  done
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

config () {  # $1 = config name, $2.. = extra flags
  cfg=$1; shift
  for s in 0 1 2; do
    name=C_${cfg}_seed$s
    python -u train.py $BASE_FLAGS "$@" --seed $s --out-dir runs/$name 2>&1 | tee train_$name.log
  done
  for s in 0 1 2; do
    name=C_${cfg}_seed$s
    python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
    grep -q "map_transform=mirror, split_version=v2" eval_${name}_e12.log \
      || { echo "eval_${name}_e12.log: not evaluated with mirror / v2"; exit 1; }
  done
  echo "== C_${cfg}: ALL rows (tuning split v2)"; grep -h "^ALL " eval_C_${cfg}_seed?_e12.log
}

config Dnomap --no-map
config EgoOnly --no-cpm --no-coop
config D

FULL=C_D_seed0,C_D_seed1,C_D_seed2
python analyze_seeds.py --rule map --full $FULL --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2 2>&1 \
  | tee analyze_seeds_corrected_map.log
python analyze_seeds.py --rule component --full $FULL --nomap C_EgoOnly_seed0,C_EgoOnly_seed1,C_EgoOnly_seed2 2>&1 \
  | tee analyze_seeds_corrected_v2x.log
python analyze_conditions.py --rule corrected 2>&1 | tee analyze_conditions_corrected.log
echo "CORRECTED RERUN DONE"
