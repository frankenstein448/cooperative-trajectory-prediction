#!/bin/bash
# runs/DECISION_RULES_FINAL_TEST.md: final test evaluation on split v2. 18 epoch-12 checkpoints, each evaluated
# ONCE on the test split (nothing retrained):
#   Part 1: C_D / C_Dnomap / C_EgoOnly seeds 0-2: eval_scenes.py --split test --confirm-final-test --final-test
#           -> eval_<run>_test_final.log
#   Part 2: E_EgoOnly / E_V2X / E_V2Xnomap seeds 0-2: eval_env.py --split test --confirm-final-test
#           -> eval_env_<run>_test_final.log, runs/<run>/eval_env_test_checkpoint_epoch12.npz
# Each evaluation is logged in runs/TEST_EVALS.log ("final test evaluation (split v2)") before it evaluates.
# Re-runs only after a crash with no result printed (final_test_guard.py, enforced by the eval scripts).
#   ./run_final_test.sh                first launch: refuses if any test output of these runs exists
#   ./run_final_test.sh --after-crash  after a crash: accepts only this study's own logs / entries, skips every
#                                      checkpoint whose log has result lines; the guard decides for the rest
set -euo pipefail
AFTER_CRASH=0
case "${1:-}" in
  "") ;;
  --after-crash) AFTER_CRASH=1 ;;
  *) echo "usage: $0 [--after-crash]"; exit 1 ;;
esac
P1="C_D_seed0 C_D_seed1 C_D_seed2 C_Dnomap_seed0 C_Dnomap_seed1 C_Dnomap_seed2 C_EgoOnly_seed0 C_EgoOnly_seed1 C_EgoOnly_seed2"
P2="E_EgoOnly_seed0 E_EgoOnly_seed1 E_EgoOnly_seed2 E_V2X_seed0 E_V2X_seed1 E_V2X_seed2 E_V2Xnomap_seed0 E_V2Xnomap_seed1 E_V2Xnomap_seed2"
PURPOSE="final test evaluation (split v2)"
OUTPUTS="analyze_seeds_final_test_map.log analyze_seeds_final_test_v2x.log analyze_conditions_final_test.log analyze_env_final_test.log"

# ---------- preflight ----------
# everything the evaluation depends on is committed (every TEST_EVALS.log entry records git rev-parse HEAD)
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_FINAL_TEST.md \
           runs/DECISION_RULES_CORRECTED.md runs/DECISION_RULES_ENV.md run_final_test.sh)" ]; then
  echo "uncommitted changes to code / split / lanes / rules / this script; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch runs/DECISION_RULES_FINAL_TEST.md runs/DECISION_RULES_CORRECTED.md runs/DECISION_RULES_ENV.md \
  test_scenes_v2.json run_final_test.sh final_test_guard.py > /dev/null \
  || { echo "rules / split / script / guard not committed"; exit 1; }
for n in $P1 $P2; do
  [ -f runs/$n/checkpoint_epoch12.pt ] || { echo "missing runs/$n/checkpoint_epoch12.pt"; exit 1; }
  python3 -c "import json,sys; c=json.load(open('runs/$n/config.json')); \
sys.exit(0 if (c.get('split_version'), c.get('map_transform')) == ('v2', 'mirror') else 1)" \
    || { echo "runs/$n: not a split v2 / mirror run"; exit 1; }
done
if [ $AFTER_CRASH -eq 0 ]; then  # (after a crash the analyses, deterministic on the logs, are simply recomputed)
  for f in $OUTPUTS; do [ ! -e "$f" ] || { echo "$f exists; refusing"; exit 1; }; done
fi
exists () { for p in "$@"; do compgen -G "$p" > /dev/null && return 0; done; return 1; }
for n in $P1 $P2; do
  # test output of these runs other than this study's own logs: never allowed
  other=$(compgen -G "eval_${n}_test*.log"; compgen -G "eval_env_${n}_test*.log"; true)
  other=$(printf '%s\n' $other | grep -vE "^eval_(env_)?${n}_test_final(_attempt[0-9]+)?\.log$" || true)
  [ -z "$other" ] || { echo "other test-split logs exist for $n: $other"; exit 1; }
  if grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log | grep -vqP "\tpurpose=$(printf '%s' "$PURPOSE" | sed 's/[()]/\\&/g')"; then
    echo "runs/TEST_EVALS.log has a test evaluation of runs/$n with another purpose; refusing"; exit 1
  fi
  if [ $AFTER_CRASH -eq 0 ]; then
    if exists "eval_${n}_test_final*.log" "eval_env_${n}_test_final*.log" "runs/$n/eval_env_test_*.npz" \
       || grep -qP "\trun_dir=runs/$n\t" runs/TEST_EVALS.log; then
      echo "test output or a TEST_EVALS.log entry exists for $n; a first launch refuses (after a crash: --after-crash)"; exit 1
    fi
  fi
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

status () {  # $1 log, $2 run, $3 scenes|env -> fresh | rerun | final: <reason>
  python final_test_guard.py "$1" "runs/$2" "runs/$2/checkpoint_epoch12.pt" "$3"
}

# ---------- evaluations ----------
for n in $P1; do
  s=$(status eval_${n}_test_final.log $n scenes)
  if [[ $s == final* ]]; then echo "$n: $s -- not re-run"; continue; fi
  echo "== $n ($s)"
  python -u eval_scenes.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt --split test --confirm-final-test --final-test  # eval_scenes.py also writes its stdout and stderr to eval_${n}_test_final.log
done
for n in $P2; do
  s=$(status eval_env_${n}_test_final.log $n env)
  if [[ $s == final* ]]; then echo "$n: $s -- not re-run"; continue; fi
  echo "== $n ($s)"
  python -u eval_env.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt --split test --confirm-final-test  # also written to its log
done

# ---------- completeness ----------
bad=0
for n in $P1; do
  f=eval_${n}_test_final.log
  { grep -q "map_transform=mirror, split_version=v2" $f && grep -q "Evaluating on the TEST split" $f \
    && grep -q "^ALL " $f && grep -q "^COND ALL" $f; } || { echo "INCOMPLETE: $f"; bad=1; }
done
for n in $P2; do
  f=eval_env_${n}_test_final.log
  { grep -q "map_transform=mirror, split_version=v2" $f && grep -q "Evaluating on the TEST split" $f \
    && grep -q "^wrote runs/$n/eval_env_test_checkpoint_epoch12.npz" $f && [ -f runs/$n/eval_env_test_checkpoint_epoch12.npz ]; } \
    || { echo "INCOMPLETE: $f"; bad=1; }
done
echo "runs/TEST_EVALS.log entries of this study:"; grep -F "purpose=$PURPOSE" runs/TEST_EVALS.log || true
compgen -G "eval_*_test_final_attempt*.log" | sed 's/^/attempt log: /' || true
[ $bad -eq 0 ] || { echo "incomplete evaluations (see above); stopping before the analyses"; exit 1; }

# ---------- analyses (runs/DECISION_RULES_FINAL_TEST.md) ----------
FULL=C_D_seed0,C_D_seed1,C_D_seed2
python analyze_seeds.py --rule map --split test --suffix test_final --full $FULL \
  --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2 2>&1 | tee analyze_seeds_final_test_map.log
python analyze_seeds.py --rule component --split test --suffix test_final --full $FULL \
  --nomap C_EgoOnly_seed0,C_EgoOnly_seed1,C_EgoOnly_seed2 2>&1 | tee analyze_seeds_final_test_v2x.log
python analyze_conditions.py --rule corrected --split test --suffix test_final 2>&1 | tee analyze_conditions_final_test.log
python analyze_env.py --split test 2>&1 | tee analyze_env_final_test.log
echo "FINAL TEST DONE"
