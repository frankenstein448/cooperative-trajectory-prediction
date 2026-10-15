#!/bin/bash
# runs/DECISION_RULES_RSU.md, "Test evaluation": the one guarded test evaluation of E_V2X+RSU. The 3 epoch-12
# checkpoints E_V2XRSU_seed{0,1,2} are each evaluated ONCE on the split v2 test split (nothing retrained):
#   eval_env.py --split test --confirm-final-test -> eval_env_<run>_test_final.log, runs/<run>/eval_env_test_checkpoint_epoch12.npz
# then analyze_rsu.py --split test -> analyze_rsu_final_test.log.
# E_V2X seeds 0-2 are NOT re-evaluated: analyze_rsu.py reads their final-test npz (runs/DECISION_RULES_FINAL_TEST.md,
# commit f322abb). Same code for E_V2X: the RSU extension leaves use_rsu-off batches, outputs and init unchanged
# (verify_rsu_off.log, RESULT: PASS), the same reuse the tuning analysis made.
# Each evaluation is logged in runs/TEST_EVALS.log ("final test evaluation (split v2)", eval_env.py's purpose) before it
# evaluates. Re-runs only after a crash with no result printed (final_test_guard.py, enforced by eval_env.py).
#   ./run_rsu_final_test.sh                first launch: refuses if any test output of these runs exists
#   ./run_rsu_final_test.sh --after-crash  after a crash: accepts only this study's own logs / entries, skips every
#                                          checkpoint whose log has result lines; the guard decides for the rest
set -euo pipefail
AFTER_CRASH=0
case "${1:-}" in
  "") ;;
  --after-crash) AFTER_CRASH=1 ;;
  *) echo "usage: $0 [--after-crash]"; exit 1 ;;
esac
RUNS="E_V2XRSU_seed0 E_V2XRSU_seed1 E_V2XRSU_seed2"
REF="E_V2X_seed0 E_V2X_seed1 E_V2X_seed2"
PURPOSE="final test evaluation (split v2)"
OUTPUTS="analyze_rsu_final_test.log"

# ---------- preflight ----------
# everything the evaluation depends on is committed (every TEST_EVALS.log entry records git rev-parse HEAD)
if [ -n "$(git status --porcelain -- '*.py' test_scenes_v2.json 'lane_polylines_*.npz' runs/DECISION_RULES_RSU.md \
           runs/DECISION_RULES_FINAL_TEST.md runs/DECISION_RULES_ENV.md run_rsu_final_test.sh)" ]; then
  echo "uncommitted changes to code / split / lanes / rules / this script; commit first"; git status --short; exit 1
fi
git ls-files --error-unmatch runs/DECISION_RULES_RSU.md runs/DECISION_RULES_FINAL_TEST.md runs/DECISION_RULES_ENV.md \
  test_scenes_v2.json run_rsu_final_test.sh final_test_guard.py verify_rsu_off.log > /dev/null \
  || { echo "rules / split / script / guard / verify log not committed"; exit 1; }
grep -q "RESULT: PASS" verify_rsu_off.log || { echo "verify_rsu_off.log does not say RESULT: PASS"; exit 1; }
# the rules allow the test only after all tuning verdicts are written into the rules file
for h in H1 H2 H3; do
  grep -q "VERDICT $h:" runs/DECISION_RULES_RSU.md || { echo "runs/DECISION_RULES_RSU.md has no tuning VERDICT $h"; exit 1; }
done
for n in $RUNS; do
  [ -f runs/$n/checkpoint_epoch12.pt ] || { echo "missing runs/$n/checkpoint_epoch12.pt"; exit 1; }
  python3 -c "import json,sys; c=json.load(open('runs/$n/config.json')); \
sys.exit(0 if (c.get('split_version'), c.get('map_transform'), c.get('v2x'), c.get('no_map'), c.get('use_rsu')) \
== ('v2', 'mirror', 'on', False, True) else 1)" \
    || { echo "runs/$n: not a split v2 / mirror / v2x on / map / use_rsu run"; exit 1; }
done
# reused E_V2X test evaluations: complete, one final-test entry each, nothing else on the test split
for n in $REF; do
  f=eval_env_${n}_test_final.log
  { [ -f runs/$n/eval_env_test_checkpoint_epoch12.npz ] && grep -q "^wrote runs/$n/eval_env_test_checkpoint_epoch12.npz" $f \
    && grep -q "map_transform=mirror, split_version=v2" $f && grep -q "Evaluating on the TEST split" $f; } \
    || { echo "reused E_V2X test evaluation incomplete or missing: runs/$n / $f"; exit 1; }
  [ "$(grep -cP "\trun_dir=runs/$n\t" runs/TEST_EVALS.log)" -eq 1 ] \
    || { echo "runs/TEST_EVALS.log: expected exactly one test entry for runs/$n"; exit 1; }
  grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log | grep -qF "purpose=$PURPOSE" \
    || { echo "runs/TEST_EVALS.log: the runs/$n entry is not \"$PURPOSE\""; exit 1; }
done
if [ $AFTER_CRASH -eq 0 ]; then  # (after a crash the analysis, deterministic on the npz files, is simply recomputed)
  for f in $OUTPUTS; do [ ! -e "$f" ] || { echo "$f exists; refusing"; exit 1; }; done
fi
exists () { for p in "$@"; do compgen -G "$p" > /dev/null && return 0; done; return 1; }
for n in $RUNS; do
  # test output of these runs other than this study's own logs: never allowed
  other=$(compgen -G "eval_${n}_test*.log"; compgen -G "eval_env_${n}_test*.log"; true)
  other=$(printf '%s\n' $other | grep -vE "^eval_env_${n}_test_final(_attempt[0-9]+)?\.log$" || true)
  [ -z "$other" ] || { echo "other test-split logs exist for $n: $other"; exit 1; }
  if grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log | grep -vqP "\tpurpose=$(printf '%s' "$PURPOSE" | sed 's/[()]/\\&/g')"; then
    echo "runs/TEST_EVALS.log has a test evaluation of runs/$n with another purpose; refusing"; exit 1
  fi
  if [ $AFTER_CRASH -eq 0 ]; then
    if exists "eval_env_${n}_test_final*.log" "runs/$n/eval_env_test_*.npz" \
       || grep -qP "\trun_dir=runs/$n\t" runs/TEST_EVALS.log; then
      echo "test output or a TEST_EVALS.log entry exists for $n; a first launch refuses (after a crash: --after-crash)"; exit 1
    fi
  fi
done

source ~/gpu-test-venv/bin/activate
ulimit -n 65535

status () {  # $1 log, $2 run -> fresh | rerun | final: <reason>
  python final_test_guard.py "$1" "runs/$2" "runs/$2/checkpoint_epoch12.pt" env
}

# ---------- evaluations ----------
for n in $RUNS; do
  s=$(status eval_env_${n}_test_final.log $n)
  if [[ $s == final* ]]; then echo "$n: $s -- not re-run"; continue; fi
  echo "== $n ($s)"
  python -u eval_env.py --run-dir runs/$n --ckpt runs/$n/checkpoint_epoch12.pt --split test --confirm-final-test  # also written to its log
done

# ---------- completeness ----------
bad=0
for n in $RUNS; do
  f=eval_env_${n}_test_final.log
  { grep -q "map_transform=mirror, split_version=v2, use_rsu=True" $f && grep -q "Evaluating on the TEST split" $f \
    && grep -q "^wrote runs/$n/eval_env_test_checkpoint_epoch12.npz" $f && [ -f runs/$n/eval_env_test_checkpoint_epoch12.npz ]; } \
    || { echo "INCOMPLETE: $f"; bad=1; }
done
echo "runs/TEST_EVALS.log entries of this study:"
for n in $RUNS; do grep -P "\trun_dir=runs/$n\t" runs/TEST_EVALS.log || true; done
compgen -G "eval_env_E_V2XRSU_seed*_test_final_attempt*.log" | sed 's/^/attempt log: /' || true
[ $bad -eq 0 ] || { echo "incomplete evaluations (see above); stopping before the analysis"; exit 1; }

# ---------- analysis (runs/DECISION_RULES_RSU.md; E_V2X from the final test) ----------
python analyze_rsu.py --split test 2>&1 | tee analyze_rsu_final_test.log
echo "RSU FINAL TEST DONE"
