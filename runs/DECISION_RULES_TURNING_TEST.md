# Decision rule: confirmatory test of "the map lowers minADE6 on turning vehicles"

Written and committed **before any condition-based number was computed on the test split**.
Evaluation only: no training, no model change. The existing epoch-12 checkpoints are evaluated
once more on the **test split**.

## Origin of the hypothesis (tuning split, exploratory)

From the condition-based evaluation (`runs/DECISION_RULES_CONDITIONS.md`, `analyze_conditions.log`),
map comparison, turning condition, pooled tuning split (n = 279):

    D 8.177 +/- 0.240   Dnomap 8.599 +/- 0.245   gap (Dnomap - D) +0.422 m   threshold 2 x 0.245 = 0.491 m

D was lower, but the gap was below the threshold, so the tuning verdict was "no detectable effect".
It was the largest map gap of the four conditions relative to its threshold, and it was picked from
4 conditions × 2 comparisons after seeing the tuning numbers. This study tests that one hypothesis
on held-out data.

## Hypothesis

**The map lowers minADE6 on turning vehicles.** One hypothesis, one comparison, one direction.

## Runs (existing checkpoints, `checkpoint_epoch12.pt`, unchanged)

- `D_seed0..2`: full model (map + V2X)
- `Dnomap_seed0..2`: D without the map

These six checkpoints were already evaluated on the test split (`eval_<run>_test.log`, overall and
per-scene rows; `eval_<run>_test_safety.log`, a metrics-only re-run). The test split is therefore not
untouched for these models, but no condition label or condition-based number has been computed on it.

## Evaluation (`run_turning_test.sh`)

Each checkpoint has **one evaluation that produces condition results**:
`eval_scenes.py --split test --confirm-final-test --confirmatory-turning-test`, which writes its own
stdout and stderr to `eval_<run>_test_cond.log`.

- Each evaluation is appended to `runs/TEST_EVALS.log` before it evaluates, with
  `purpose=confirmatory condition evaluation (turning hypothesis from tuning)`.
- **Re-run after a crash** (amendment, committed before any launch): a re-run for a checkpoint is
  allowed only if **every** earlier attempt's log contains **no line starting with `COND`**, i.e. the
  attempt crashed before any condition result existed. `eval_scenes.py` enforces this:
  - The earlier attempts are the existing `eval_<run>_test_cond.log` and every
    `eval_<run>_test_cond_attempt<N>.log`. If `runs/TEST_EVALS.log` has more confirmatory entries for the
    checkpoint than there are attempt logs, the re-run is refused (a missing log cannot be checked).
  - Any attempt log with a `COND` line makes that attempt **final**: the re-run is refused. This includes
    the `CONDITIONS` header line that precedes the condition rows, so an attempt that got as far as
    the condition block is final even if no row was printed.
  - On an allowed re-run, the earlier `eval_<run>_test_cond.log` is kept, renamed
    `eval_<run>_test_cond_attempt<N>.log` (N = 1, 2, … in order of the attempts), and the re-run is
    logged in `runs/TEST_EVALS.log` with
    `purpose=confirmatory condition evaluation (turning hypothesis from tuning) (re-run after crash, no results seen)`.
  - An attempt that crashed before its `runs/TEST_EVALS.log` entry was written (e.g. while loading the
    data) still has a log and counts as an attempt; a re-run after it is also marked as a re-run.
- `run_turning_test.sh` skips a checkpoint whose `eval_<run>_test_cond.log` already has `COND` lines,
  so re-launching it after a crash re-runs only the checkpoints the rule allows. All attempt logs and
  `runs/TEST_EVALS.log` entries are reported.
- Each run's accuracy rows (per-scene and ALL) must match its original `eval_<run>_test.log` exactly
  (same checkpoint and data). Any difference stops the script before the analysis, and is reported.

Condition labels (turning, straight, stationary, near_junction, and the complements moving /
not_near_junction) are exactly those of `runs/DECISION_RULES_CONDITIONS.md` (`scene_conditions.py`,
unchanged: turning = moving and |heading change over the 30-step GT| > 30°). They depend on ground
truth only, so every condition's n must be identical across the six logs.

## Rule (`analyze_conditions.py --rule turning_test` → `analyze_turning_test.log`)

With seed means *m* and sample standard deviations *s* (n = 3 seeds) of minADE6 on the **turning**
subset, pooled over the test split (the `COND ALL turning` row):

- **Confirmed** only if *m*(D) < *m*(Dnomap) **and** *m*(Dnomap) − *m*(D) > 2 × max(*s*(D), *s*(Dnomap)).
- Otherwise: **not confirmed** (no detectable effect).

The same rule as on tuning. A single pre-registered hypothesis, so no multiple-comparison correction.
If the test split has no turning sample, the result is "not evaluable".

## Sanity checks (as in `analyze_conditions.py`; any failure stops it)

- The ALL row has n = 2,208 in every log.
- Every condition's n is identical across the six logs.
- stationary + moving = ALL, turning + straight = moving, near + not-near = ALL.
- COND stationary n equals the existing `stationary (ALL)` n in the same log.

## Reporting

Reported whatever the outcome: the turning n, per-run and seed mean ± std of minADE6 (rule) and of
ADE1 and minFDE6 (descriptive). The other conditions (straight, stationary, near_junction, moving,
not_near_junction) are reported with their n, seed means, stds and the same gap and threshold, as
**descriptive only**: no verdict is drawn from them. Per-town rows are descriptive.
