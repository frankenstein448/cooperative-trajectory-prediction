# Decision rules: final test evaluation (split v2)

Written and committed **before any checkpoint of this study was evaluated on the split v2 test split**.
Evaluation only: no training, no model change, no checkpoint choice. Each checkpoint below is evaluated
**once** on the test split; nothing learned from these numbers feeds back into any model or rule.

## Status when this document was committed (2026-10-14)

- `runs/TEST_EVALS.log` has no entry for any `C_*` or `E_*` run, and no `eval_*_test*.log` /
  `eval_env_test_*.npz` exists for them: the split v2 test split is untouched by these 18 checkpoints.
- Test split v2 (`test_scenes_v2.json`): Town03 5; Town04 6; Town05 76, 78, 89, 91, 95, 100 (8 scenes).
  Part 1: 2,208 samples. Part 2: 1,976 env samples (D's samples with at least one class-(a) target). Scene 78
  was in Part 1's v1 test split (`DECISION_RULES_CORRECTED.md`); no model choice depended on it.
- The tuning verdicts of both studies were known when this document was written (listed below for the
  side-by-side report). The rules are not changed in light of them: they are the tuning rules, verbatim.

## Runs (`checkpoint_epoch12.pt`, unchanged)

| study | runs | rules |
|---|---|---|
| Part 1, corrected rerun | `C_D_seed0..2`, `C_Dnomap_seed0..2`, `C_EgoOnly_seed0..2` | `runs/DECISION_RULES_CORRECTED.md` |
| Part 2, env prediction | `E_EgoOnly_seed0..2`, `E_V2X_seed0..2`, `E_V2Xnomap_seed0..2` | `runs/DECISION_RULES_ENV.md` |

All 18 were trained with split v2 and map transform mirror (`config.json`); the evaluation scripts refuse any
run that was not.

## Evaluation (`run_final_test.sh`, launched by the user)

- Part 1: `eval_scenes.py --split test --confirm-final-test --final-test` → `eval_<run>_test_final.log`
  (the script writes its own stdout and stderr there). Same metrics and blocks as on tuning: per-scene and ALL
  rows, CV / stay-put, SAFE (miss, off-road, GT off-road control), RANK, COND.
- Part 2: `eval_env.py --split test --confirm-final-test` → `eval_env_<run>_test_final.log` and
  `runs/<run>/eval_env_test_checkpoint_epoch12.npz`. Same ENV / COV tables as on tuning.
- Without `--final-test`, `eval_scenes.py` refuses the test split of every split v2 run; `eval_env.py` refuses
  `--split test` without `--confirm-final-test`, and never overwrites an existing test npz.

### One evaluation per checkpoint; logging; re-runs (`final_test_guard.py`)

- Each evaluation is appended to `runs/TEST_EVALS.log` **before** it evaluates (after the data is loaded), with
  `purpose=final test evaluation (split v2)` and `split_version=v2`.
- **Re-run after a crash** (same rule as the turning test, `DECISION_RULES_TURNING_TEST.md`): a re-run of a
  checkpoint is allowed only if **every** earlier attempt's log contains **no result line**, i.e. every attempt
  crashed before any number derived from predictions was printed.
  - Result lines: `eval_scenes.py`: per-scene / ALL / CV rows, the `Sanity` lines, `Median`, `Scenes`, and every
    `SAFE` / `RANK` / `COND` line. `eval_env.py`: every `ENV` / `COV` line.
  - The earlier attempts are the existing log and every `<log stem>_attempt<N>.log`. If `runs/TEST_EVALS.log` has
    more final-test entries for the checkpoint than there are attempt logs, the re-run is refused (a missing log
    cannot be checked).
  - On an allowed re-run the earlier log is kept as `<log stem>_attempt<N>.log` (N = 1, 2, … in order) and the
    entry is marked `(re-run after crash, no results seen)`. An attempt that crashed before its entry was
    written still has a log and counts as an attempt.
- `run_final_test.sh` refuses to start if anything of these runs on the test split already exists. After a
  crash it is re-launched with `--after-crash`: it then accepts only this study's own logs and entries, skips
  every checkpoint whose log already has result lines, and leaves the decision for the others to the guard.
  The guard rule stands even then: an attempt with any result line is final, complete or not.
- Before the analyses, every log must be complete (Part 1: ALL row and COND block; Part 2: the npz written) and
  must say `map_transform=mirror, split_version=v2` and `TEST split`. Otherwise the script stops and reports.

## Rules (the tuning rules, applied to the 8 test scenes)

Seed means and sample standard deviations over seeds 0, 1, 2; lower is better.

**Part 1** (`DECISION_RULES_CORRECTED.md`):

1. **Map** (`analyze_seeds.py --rule map --split test --suffix test_final`): the map helps only if C_D's
   seed-mean minADE6 is lower than C_Dnomap's in **≥ 7 of the 8 test scenes** AND (C_Dnomap − C_D) on ALL is
   **> 2 × the larger of the two seed standard deviations**.
2. **V2X** (`--rule component`, ablation = C_EgoOnly): the same rule.
3. **Secondary, turning** (`analyze_conditions.py --rule corrected --split test --suffix test_final`): the map
   lowers minADE6 on turning vehicles only if C_D's seed-mean turning minADE6 < C_Dnomap's AND the gap >
   2 × max(seed std), pooled over the test split (`COND ALL turning`). No turning sample → "not evaluable".

**Part 2** (`DECISION_RULES_ENV.md`, `analyze_env.py --split test`), MOVING targets; full form = seed-mean lower
in ≥ 7 of 8 test scenes AND ALL gap > 2 × the larger seed std; pooled form if a test scene has no target of the
subset (decided by the ground-truth labels alone) = seed-mean ALL lower AND the same gap condition:

1. **V2X on (a)**: E_V2X vs E_EgoOnly, minADE6 on moving class-(a) targets.
2. **V2X on (b)**: E_V2X's ADE1 vs the constant-velocity baseline's ADE on moving class-(b) targets (baseline seed
   std 0). E_V2X minADE6 vs the baseline: descriptive only.
3. **Map**: E_V2X vs E_V2Xnomap, minADE6 on moving targets ((a) + (b)). **3b, secondary**: the same on turning
   targets.

One rule per question, as on tuning; no multiple-comparison correction (none was applied on tuning either).

## Sanity checks (in the evaluation and analysis scripts; any failure stops them)

- Part 1: ALL n = 2,208 in every log; 8 scenes, the same in every log; condition n identical across the nine
  logs (labels come from ground truth); stationary + moving = ALL, turning + straight = moving,
  near + not-near = ALL.
- Part 2: every npz says split test, v2, mirror, with the same scenes; each config's seeds have identical target
  lists and labels; E_EgoOnly's targets are exactly E_V2X's class-(a) targets; the evaluated target count equals
  the coverage count; the coverage is identical across seeds.

## Reported descriptively (no verdict), as on tuning

Part 1: ADE1, minFDE6, per-scene and per-town rows; the other conditions; safety (miss, off-road with the GT
control per town); ranking. Part 2: every metric × subset row, miss rates, CV and stay-put baselines,
stationary / straight / category rows, coverage per model.

## Reporting

All verdicts are reported whatever they say, next to the tuning verdicts:

| rule | tuning verdict (split v2) |
|---|---|
| Part 1 map | no detectable effect (6/8 scenes; gap −0.254 m vs threshold 0.439 m) |
| Part 1 V2X | no detectable effect (2/8 scenes; gap −0.143 m vs threshold 0.133 m) |
| Part 1 map on turning (secondary) | no detectable effect (gap +0.086 m vs threshold 2.029 m; n = 144) |
| Env (1) V2X on (a) | no detectable effect |
| Env (2) V2X on (b) vs constant velocity | better |
| Env (3) map, moving | helps |
| Env (3b) map, turning (secondary) | helps |

A test verdict that disagrees with its tuning verdict is reported as such; neither is dropped. After this
evaluation these checkpoints are not evaluated on the test split again (except the crash re-runs above).

## Commands (`run_final_test.sh`, after all 18 evaluations)

```
python analyze_seeds.py --rule map --split test --suffix test_final --full C_D_seed0,C_D_seed1,C_D_seed2 \
    --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2                  > analyze_seeds_final_test_map.log
python analyze_seeds.py --rule component --split test --suffix test_final --full C_D_seed0,C_D_seed1,C_D_seed2 \
    --nomap C_EgoOnly_seed0,C_EgoOnly_seed1,C_EgoOnly_seed2               > analyze_seeds_final_test_v2x.log
python analyze_conditions.py --rule corrected --split test --suffix test_final > analyze_conditions_final_test.log
python analyze_env.py --split test                                      > analyze_env_final_test.log
```
