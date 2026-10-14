#!/bin/bash
# runs/DECISION_RULES_RSU.md: E_V2X+RSU (run dirs E_V2XRSU_seed{0,1,2}), split v2, map transform mirror, 12 epochs,
# checkpoint = epoch 12 (no best-epoch selection). Flags = E_V2X's in run_env_seeds.sh plus --use-rsu, nothing else.
# After the three runs: eval_env.py on epoch 12 (tuning split v2) -> eval_env_<run>_e12.log and
# runs/<run>/eval_env_tuning_checkpoint_epoch12.npz; then analyze_rsu.py (H1-H3 vs the existing E_V2X evaluations)
# -> analyze_rsu.log. Never touches the test split.
#   ./run_rsu_seeds.sh
set -euo pipefail
BASE_FLAGS="--task env --epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --map-transform mirror --split-version v2"
grep -q "^BASE_FLAGS=\"$BASE_FLAGS\"$" run_env_seeds.sh || { echo "BASE_FLAGS differ from run_env_seeds.sh"; exit 1; }

# preflight: everything the runs depend on is committed and verified
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_RSU.md \
           runs/DECISION_RULES_ENV.md run_rsu_seeds.sh)" ]; then
  echo "uncommitted changes to tracked code / split / lanes / rules / this script; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch test_scenes_v2.json runs/DECISION_RULES_RSU.md runs/DECISION_RULES_ENV.md run_rsu_seeds.sh \
  lane_polylines_town03.npz lane_polylines_town04.npz lane_polylines_town05.npz verify_env_model.log \
  verify_rsu_image_cache.log verify_rsu_off.log > /dev/null \
  || { echo "rules / split / lanes / script / verify logs not committed"; exit 1; }
for log in verify_env_model.log verify_rsu_image_cache.log verify_rsu_off.log; do
  grep -q "RESULT: PASS" $log || { echo "$log does not say RESULT: PASS"; exit 1; }
done
python3 -c "import json,os,sys; m=json.load(open(os.path.expanduser('~/hd_cache/env_targets/manifest.json'))); \
sys.exit(0 if m['builder_dirty'] is False else 1)" || { echo "env cache manifest: builder_dirty is not False"; exit 1; }
python3 -c "import json,os,sys; d=os.path.expanduser('~/hd_cache/images_rsu'); m=json.load(open(d+'/manifest.json')); \
n=len([f for f in os.listdir(d) if f.endswith('.npy')]); \
sys.exit(0 if (m['file_count'] == n and m['shape'] == [4, 3, 224, 224] and m['dtype'] == 'uint8') else 1)" \
  || { echo "RSU image cache manifest missing or inconsistent (build_rsu_image_cache.py)"; exit 1; }
for s in 0 1 2; do
  [ ! -e runs/E_V2XRSU_seed$s ] || { echo "runs/E_V2XRSU_seed$s exists; refusing to overwrite"; exit 1; }
  [ -f runs/E_V2X_seed$s/eval_env_tuning_checkpoint_epoch12.npz ] \
    || { echo "missing runs/E_V2X_seed$s/eval_env_tuning_checkpoint_epoch12.npz (analyze_rsu.py needs it)"; exit 1; }
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

for s in 0 1 2; do
  name=E_V2XRSU_seed$s
  python -u train.py $BASE_FLAGS --v2x on --use-rsu --seed $s --out-dir runs/$name 2>&1 | tee train_$name.log
done
for s in 0 1 2; do
  name=E_V2XRSU_seed$s
  python -u eval_env.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_env_${name}_e12.log
  grep -q "map_transform=mirror, split_version=v2, use_rsu=True" eval_env_${name}_e12.log \
    || { echo "eval_env_${name}_e12.log: not evaluated with mirror / v2 / use_rsu"; exit 1; }
done
echo "== E_V2XRSU: ALL rows (tuning split v2)"; grep -h "^ENV       ALL all " eval_env_E_V2XRSU_seed?_e12.log

python analyze_rsu.py 2>&1 | tee analyze_rsu.log
echo "RSU RUNS DONE"
