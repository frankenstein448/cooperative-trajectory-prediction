#!/bin/bash
# runs/DECISION_RULES_MAPV2_EGO.md: D with map v2 (run dirs C_Dmapv2_seed{0,1,2}), split v2, map transform mirror,
# 12 epochs, checkpoint = epoch 12. Flags = C_D's in run_corrected_seeds.sh plus --map-version v2, nothing else.
# After the three runs: eval_scenes.py on epoch 12 (tuning split v2) -> eval_C_Dmapv2_seed<k>_e12.log; then the
# primary rule (analyze_seeds.py --rule map vs the existing C_Dnomap logs) -> analyze_seeds_mapv2_ego.log and the
# secondary turning rule -> analyze_conditions_mapv2_ego.log. Never touches the test split.
#   ./run_mapv2_ego_seeds.sh
set -euo pipefail
BASE_FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --map-transform mirror --split-version v2"
grep -q "^BASE_FLAGS=\"$BASE_FLAGS\"$" run_corrected_seeds.sh || { echo "BASE_FLAGS differ from run_corrected_seeds.sh"; exit 1; }
RUNS="C_Dmapv2_seed0 C_Dmapv2_seed1 C_Dmapv2_seed2"

# preflight: everything the runs depend on is committed and verified (every run records git rev-parse HEAD)
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_MAPV2_EGO.md \
           runs/DECISION_RULES_CORRECTED.md run_mapv2_ego_seeds.sh)" ]; then
  echo "uncommitted changes to tracked code / split / lanes / rules / this script; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch test_scenes_v2.json runs/DECISION_RULES_MAPV2_EGO.md runs/DECISION_RULES_CORRECTED.md \
  run_mapv2_ego_seeds.sh lane_polylines_town03.npz lane_polylines_town04.npz lane_polylines_town05.npz \
  verify_mapv2_off.log verify_mapv2_tokens.log > /dev/null \
  || { echo "rules / split / lanes / script / verify logs not committed"; exit 1; }
for log in verify_mapv2_off.log verify_mapv2_tokens.log; do
  grep -q "RESULT: PASS" $log || { echo "$log does not say RESULT: PASS"; exit 1; }
done
grep -q "Status: DRAFT" runs/DECISION_RULES_MAPV2_EGO.md && { echo "runs/DECISION_RULES_MAPV2_EGO.md is still a DRAFT"; exit 1; }
for s in 0 1 2; do
  n=C_Dmapv2_seed$s
  [ ! -e runs/$n ] || { echo "runs/$n exists; refusing to overwrite"; exit 1; }
  for ref in C_Dnomap C_D; do  # reused runs: their flags, checkpoint and tuning log
    r=${ref}_seed$s
    [ -f runs/$r/checkpoint_epoch12.pt ] && grep -q "map_transform=mirror, split_version=v2" eval_${r}_e12.log \
      || { echo "runs/$r or eval_${r}_e12.log missing / not mirror + v2"; exit 1; }
    python3 -c "import json,sys; c=json.load(open('runs/$r/config.json')); \
sys.exit(0 if (c['epochs'], c['use_image_cache'], c['use_speed'], c['normalize_cam'], c['drop_cam_xy'], c['loss'], \
c['map_transform'], c['split_version'], c['no_map'], c['no_cpm'], c['no_coop'], c['history_frames'], c['degrade'], \
c.get('map_version', 'v1')) == (12, True, True, True, True, 'wta', 'mirror', 'v2', '$ref' == 'C_Dnomap', False, \
False, 0, None, 'v1') else 1)" || { echo "runs/$r/config.json: flags differ from the setup"; exit 1; }
  done
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

for s in 0 1 2; do
  n=C_Dmapv2_seed$s
  python -u train.py $BASE_FLAGS --map-version v2 --seed $s --out-dir runs/$n 2>&1 | tee train_$n.log
done
for n in $RUNS; do
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt 2>&1 | tee eval_${n}_e12.log
  { grep -q "map_transform=mirror, split_version=v2" eval_${n}_e12.log && grep -q "map_version=v2" eval_${n}_e12.log; } \
    || { echo "eval_${n}_e12.log: not evaluated with mirror / v2 / map_version v2"; exit 1; }
done
echo "== ALL rows (tuning split v2): C_Dmapv2, C_Dnomap (rule), C_D (map v1, descriptive)"
grep -h "^ALL " eval_C_Dmapv2_seed?_e12.log eval_C_Dnomap_seed?_e12.log eval_C_D_seed?_e12.log

python analyze_seeds.py --rule map --full C_Dmapv2_seed0,C_Dmapv2_seed1,C_Dmapv2_seed2 \
  --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2 2>&1 | tee analyze_seeds_mapv2_ego.log
python analyze_conditions.py --rule corrected --map-full Dmapv2 2>&1 | tee analyze_conditions_mapv2_ego.log
echo "MAPV2 EGO RUNS DONE"
