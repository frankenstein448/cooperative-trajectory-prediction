#!/bin/bash
# runs/DECISION_RULES_MAPV2_EGO.md, "Test evaluation": the one guarded test evaluation of map v2 in the Part 1 ego
# task. The 3 epoch-12 checkpoints C_Dmapv2_seed{0,1,2} are each evaluated ONCE on the split v2 test split (nothing
# retrained):
#   eval_scenes.py --split test --confirm-final-test --final-test -> eval_<run>_test_final.log
# then both rules on the test logs:
#   analyze_seeds.py --rule map (C_Dmapv2 vs C_Dnomap)                    -> analyze_seeds_mapv2_ego_final_test.log
#   analyze_conditions.py --rule corrected --map-full Dmapv2 --split test -> analyze_conditions_mapv2_ego_final_test.log
# C_Dnomap seeds 0-2 are NOT re-evaluated: their final-test logs eval_C_Dnomap_seed<k>_test_final.log
# (runs/DECISION_RULES_FINAL_TEST.md, commit f322abb) are reused, as the rules state. Same code for C_Dnomap:
# --map-version v1 / --no-map batches, outputs and init are unchanged (verify_mapv2_off.log, RESULT: PASS).
# Each evaluation is logged in runs/TEST_EVALS.log ("final test evaluation (split v2)", eval_scenes.py's purpose)
# before it evaluates. Re-runs only after a crash with no result printed (final_test_guard.py, enforced by
# eval_scenes.py).
#   ./run_mapv2_ego_final_test.sh                first launch: refuses if any test output of these runs exists
#   ./run_mapv2_ego_final_test.sh --after-crash  after a crash: accepts only this study's own logs / entries, skips
#                                                every checkpoint whose log has result lines; the guard decides for the rest
set -euo pipefail
AFTER_CRASH=0
case "${1:-}" in
  "") ;;
  --after-crash) AFTER_CRASH=1 ;;
  *) echo "usage: $0 [--after-crash]"; exit 1 ;;
esac
RUNS="C_Dmapv2_seed0 C_Dmapv2_seed1 C_Dmapv2_seed2"
REF="C_Dnomap_seed0 C_Dnomap_seed1 C_Dnomap_seed2"
PURPOSE="final test evaluation (split v2)"
OUTPUTS="analyze_seeds_mapv2_ego_final_test.log analyze_conditions_mapv2_ego_final_test.log"

# ---------- preflight ----------
# everything the evaluation depends on is committed (every TEST_EVALS.log entry records git rev-parse HEAD)
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_MAPV2_EGO.md \
           runs/DECISION_RULES_FINAL_TEST.md runs/DECISION_RULES_CORRECTED.md run_mapv2_ego_final_test.sh)" ]; then
  echo "uncommitted changes to code / split / lanes / rules / this script; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch runs/DECISION_RULES_MAPV2_EGO.md runs/DECISION_RULES_FINAL_TEST.md \
  runs/DECISION_RULES_CORRECTED.md test_scenes_v2.json run_mapv2_ego_final_test.sh final_test_guard.py \
  verify_mapv2_off.log verify_mapv2_tokens.log > /dev/null \
  || { echo "rules / split / script / guard / verify logs not committed"; exit 1; }
for v in verify_mapv2_off.log verify_mapv2_tokens.log; do
  grep -q "RESULT: PASS" $v || { echo "$v does not say RESULT: PASS"; exit 1; }
done
# the rules allow the test only after both tuning verdicts are written into the rules file
for h in primary secondary; do
  grep -q "VERDICT $h:" runs/DECISION_RULES_MAPV2_EGO.md \
    || { echo "runs/DECISION_RULES_MAPV2_EGO.md has no tuning VERDICT $h"; exit 1; }
done
for n in $RUNS; do
  [ -f runs/$n/checkpoint_epoch12.pt ] || { echo "missing runs/$n/checkpoint_epoch12.pt"; exit 1; }
  python3 -c "import json,sys; c=json.load(open('runs/$n/config.json')); \
sys.exit(0 if (c.get('split_version'), c.get('map_transform'), c.get('map_version'), c.get('no_map')) \
== ('v2', 'mirror', 'v2', False) else 1)" \
    || { echo "runs/$n: not a split v2 / mirror / map v2 run"; exit 1; }
done
# reused C_Dnomap test evaluations: complete, one final-test entry each, nothing else on the test split
for n in $REF; do
  f=eval_${n}_test_final.log
  { [ -f $f ] && grep -q "map_transform=mirror, split_version=v2" $f && grep -q "Evaluating on the TEST split" $f \
    && grep -q "^ALL " $f && grep -q "^COND ALL" $f && ! grep -q "^map_version=" $f; } \
    || { echo "reused C_Dnomap test evaluation incomplete or missing: $f"; exit 1; }
  [ "$(grep -cP "\trun_dir=runs/$n\t" runs/TEST_EVALS.log)" -eq 1 ] \
    || { echo "runs/TEST_EVALS.log: expected exactly one test entry for runs/$n"; exit 1; }
  grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log | grep -qF "purpose=$PURPOSE" \
    || { echo "runs/TEST_EVALS.log: the runs/$n entry is not \"$PURPOSE\""; exit 1; }
done
if [ $AFTER_CRASH -eq 0 ]; then  # (after a crash the analyses, deterministic on the logs, are simply recomputed)
  for f in $OUTPUTS; do [ ! -e "$f" ] || { echo "$f exists; refusing"; exit 1; }; done
fi
exists () { for p in "$@"; do compgen -G "$p" > /dev/null && return 0; done; return 1; }
for n in $RUNS; do
  # test output of these runs other than this study's own logs: never allowed
  other=$(compgen -G "eval_${n}_test*.log"; true)
  other=$(printf '%s\n' $other | grep -vE "^eval_${n}_test_final(_attempt[0-9]+)?\.log$" || true)
  [ -z "$other" ] || { echo "other test-split logs exist for $n: $other"; exit 1; }
  if grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log | grep -vqP "\tpurpose=$(printf '%s' "$PURPOSE" | sed 's/[()]/\\&/g')"; then
    echo "runs/TEST_EVALS.log has a test evaluation of runs/$n with another purpose; refusing"; exit 1
  fi
  if [ $AFTER_CRASH -eq 0 ]; then
    if exists "eval_${n}_test_final*.log" || grep -qP "\trun_dir=runs/$n\t" runs/TEST_EVALS.log; then
      echo "test output or a TEST_EVALS.log entry exists for $n; a first launch refuses (after a crash: --after-crash)"; exit 1
    fi
  fi
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

status () {  # $1 log, $2 run -> fresh | rerun | final: <reason>
  python final_test_guard.py "$1" "runs/$2" "runs/$2/checkpoint_epoch12.pt" scenes
}

# ---------- evaluations ----------
for n in $RUNS; do
  s=$(status eval_${n}_test_final.log $n)
  if [[ $s == final* ]]; then echo "$n: $s -- not re-run"; continue; fi
  echo "== $n ($s)"
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt --split test --confirm-final-test --final-test  # eval_scenes.py also writes its stdout and stderr to eval_${n}_test_final.log
done

# ---------- completeness ----------
bad=0
for n in $RUNS; do
  f=eval_${n}_test_final.log
  { grep -q "^map_version=v2 (from config.json)" $f && grep -q "map_transform=mirror, split_version=v2" $f \
    && grep -q "Evaluating on the TEST split" $f && grep -q "^ALL " $f && grep -q "^COND ALL" $f; } \
    || { echo "INCOMPLETE: $f"; bad=1; }
done
echo "runs/TEST_EVALS.log entries of this study:"
for n in $RUNS; do grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log || true; done
compgen -G "eval_C_Dmapv2_seed*_test_final_attempt*.log" | sed 's/^/attempt log: /' || true
[ $bad -eq 0 ] || { echo "incomplete evaluations (see above); stopping before the analyses"; exit 1; }

# ---------- analyses (runs/DECISION_RULES_MAPV2_EGO.md; C_Dnomap from the final test) ----------
python analyze_seeds.py --rule map --split test --suffix test_final --full C_Dmapv2_seed0,C_Dmapv2_seed1,C_Dmapv2_seed2 \
  --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2 2>&1 | tee analyze_seeds_mapv2_ego_final_test.log
python analyze_conditions.py --rule corrected --map-full Dmapv2 --split test --suffix test_final 2>&1 \
  | tee analyze_conditions_mapv2_ego_final_test.log
echo "MAP V2 EGO FINAL TEST DONE"
