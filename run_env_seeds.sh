#!/bin/bash
# runs/DECISION_RULES_ENV.md: Part 2 environment prediction, split v2, map transform mirror, seeds 0-2, 12 epochs.
# Order: E_EgoOnly x3, E_V2Xnomap x3, E_V2X x3. After each config's three runs: eval_env.py on epoch 12 (tuning
# split v2) -> eval_env_<run>_e12.log and runs/<run>/eval_env_tuning_checkpoint_epoch12.npz. At the end:
# analyze_env.py -> analyze_env.log.
#   ./run_env_seeds.sh
set -euo pipefail
BASE_FLAGS="--task env --epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --map-transform mirror --split-version v2"

# preflight: everything the runs depend on is committed and verified
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_ENV.md)" ]; then
  echo "uncommitted changes to tracked code / split / lanes / rules; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch test_scenes_v2.json runs/DECISION_RULES_ENV.md lane_polylines_town03.npz \
  lane_polylines_town04.npz lane_polylines_town05.npz verify_env_model.log > /dev/null \
  || { echo "rules / split / lanes / verify_env_model.log not committed"; exit 1; }
grep -q "RESULT: PASS" verify_env_model.log || { echo "verify_env_model.log does not say RESULT: PASS"; exit 1; }
python3 -c "import json,os,sys; m=json.load(open(os.path.expanduser('~/hd_cache/env_targets/manifest.json'))); \
sys.exit(0 if m['builder_dirty'] is False else 1)" || { echo "env cache manifest: builder_dirty is not False"; exit 1; }
for cfg in EgoOnly V2Xnomap V2X; do
  for s in 0 1 2; do
    [ ! -e runs/E_${cfg}_seed$s ] || { echo "runs/E_${cfg}_seed$s exists; refusing to overwrite"; exit 1; }
  done
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

config () {  # $1 = config name, $2.. = extra flags
  cfg=$1; shift
  for s in 0 1 2; do
    name=E_${cfg}_seed$s
    python -u train.py $BASE_FLAGS "$@" --seed $s --out-dir runs/$name 2>&1 | tee train_$name.log
  done
  for s in 0 1 2; do
    name=E_${cfg}_seed$s
    python -u eval_env.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_env_${name}_e12.log
    grep -q "map_transform=mirror, split_version=v2" eval_env_${name}_e12.log \
      || { echo "eval_env_${name}_e12.log: not evaluated with mirror / v2"; exit 1; }
  done
  echo "== E_${cfg}: ALL rows (tuning split v2)"; grep -h "^ENV       ALL all " eval_env_E_${cfg}_seed?_e12.log
}

config EgoOnly --v2x off
config V2Xnomap --v2x on --no-map
config V2X --v2x on

python analyze_env.py 2>&1 | tee analyze_env.log
echo "ENV RUNS DONE"
