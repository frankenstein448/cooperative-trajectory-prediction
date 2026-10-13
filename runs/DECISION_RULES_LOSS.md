# Decision rules: does a Laplace-mixture loss improve mode ranking?

Written and committed **before any training for this study was run**. The study uses the
**tuning split only**. The test split is not used for any part of it.

## Question

D is trained with winner-takes-all (WTA): only the closest mode is regressed, and the mode
scores learn to predict which mode was closest. Its best-of-6 error (minADE6) is far below
its top-1 error (ADE1), so a good mode usually exists but is not ranked first. Does training
with a trajectory-level Laplace-mixture likelihood lower the top-1 error (ADE1) without
giving up best-of-6 accuracy or collapsing the modes?

## Loss (fixed here, `trajectory_loss.py`)

`--loss laplace_mixture` (`LaplaceMixtureLoss`). Each mode k predicts per-step locations
mu and per-step, per-coordinate scales b. For each sample:

    log p_k = sum over valid steps t and coordinates d in {x, y} of log Laplace(gt_td; mu_ktd, b_ktd)
    NLL     = -logsumexp_k( log_softmax(scores)_k + log p_k )

The loss is the batch mean of the NLL. Invalid steps are masked as in WTA: they add 0 to
log p_k, and a sample with no valid step adds 0 and has no gradient. There is no separate
classification term, because the mode scores are the mixture weights. Top-1 is still the
argmax of the scores.

- Scales: b = softplus(raw) + 0.01 m (`LAPLACE_SCALE_FLOOR`, fixed).
- Model: the trajectory head gets one extra `Linear(128 → 6·30·2)` (`scale_mlp`,
  +46,440 parameters). Nothing else changes. With any other loss the model is D's:
  `train.py` asserts the parameter count (D's count plus `scale_mlp` for laplace_mixture,
  exactly D's count otherwise), and `verify_loss.py` checks that D's checkpoint loads with
  `strict=True`.
- The scale head is initialised from a forked RNG. So with the same `--seed`, every D
  parameter starts from the same values as in D's run, and the data order is the same.
  GPU non-determinism still applies.
- Optimiser, LR (1e-4), batch size and epochs are unchanged from D.
- `train.py` logs the NLL as `reg`. Its `cls` is the scores' cross-entropy against the
  ADE winner, which is logged for comparison with D and is not optimised. The checkpoint
  used is epoch 12, not `best_model.pt`, so the different val-loss scale does not matter.

`--loss wta_soft` (`WTASoftLoss`) is also implemented. It keeps WTA regression and trains
the scores with cross-entropy against `softmax(-ADE_k / tau)`, with tau = 1.0 m fixed.
**It is not run in this study.** It is the fallback if laplace_mixture is not adopted, and
its own decision rules will be committed before any wta_soft run.

`verify_loss.py` must pass before training. It uses random tensors only and checks finite
losses and gradients; that one mode reduces to a single Laplace NLL; a per-mode reference
implementation; that wta_soft approaches WTA's target as tau → 0; the model and parameter
checks above; and the metric implementations.

## Metrics (`eval_scenes.py`, `ranking_metrics.py`)

These come from the existing rows (ADE1, minADE6 per scene and ALL) plus a new `RANK` block:

- **Brier-minFDE**: minFDE6 + (1 − p)², where p is the softmax score of the min-FDE mode.
- **Mode diversity**: the mean pairwise distance between the 6 modes at the final horizon
  step (step 30, which is 6 s ahead), averaged over valid samples.
- **Calibration** (laplace_mixture runs only), for the top-scored mode's Laplace:
  - `pt50` / `pt90`: the fraction of valid GT points inside the box made by both
    coordinates' √L central intervals (nominal 50% / 90%).
  - `xy50` / `xy90`: the fraction of valid GT coordinates inside their L central interval.

Brier-minFDE and calibration are descriptive only.

## Runs

All runs use D's flags (`--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy`,
map on) and are evaluated at `checkpoint_epoch12.pt` on the tuning split (2,208 samples, 8 scenes).

| Run | Extra flags | Note |
|---|---|---|
| `D_seed0..2` | none | existing checkpoints, re-evaluated only for the RANK block → `eval_D_seed*_e12_rank.log`; accuracy rows must match `eval_D_seed*_e12.log` exactly or the script stops before any training |
| `D_seed3`, `D_seed4` | `--seed 3` / `--seed 4` | new runs with D's exact flags |
| `L_seed0..4` | `--seed s --loss laplace_mixture` | |

`run_loss_seeds.sh` runs these in order: D re-evaluations, D_seed3, D_seed4, L_seed0..4.
It then runs `analyze_seeds.py --rule loss --expect-n 2208 --full L_seed0,...,L_seed4
--nomap D_seed0:e12_rank,D_seed1:e12_rank,D_seed2:e12_rank,D_seed3,D_seed4`. In that output,
"full" means L and "no-map" means D.

## Rule

**laplace_mixture is adopted** only if all four of these hold (seed means over 5 seeds each):

1. L's seed-mean ALL **ADE1** is lower than D's, **and**
2. L's seed-mean **ADE1** is lower than D's in **≥ 6 of the 8** tuning scenes (a tie does
   not count as lower), **and**
3. L's seed-mean ALL **minADE6** is not worse than D's by more than 0.1 m
   (L − D ≤ +0.100 m, using the logged 3-decimal values), **and**
4. L's seed-mean ALL **mode diversity** is **≥ 50%** of D's.

The ADE1 seed std of L and D is reported. A lower std for L is the intended effect, but it
is not a criterion with 5 seeds. If any criterion fails, laplace_mixture is not adopted,
D stays the baseline, and wta_soft becomes the next candidate under its own pre-registered
rules.

## Reporting

The result is reported whatever the outcome. For every run it includes ADE1, minADE6,
minFDE6, Brier-minFDE and diversity (plus calibration for L), with seed mean ± std and the
per-scene ADE1 / minADE6 table. The SAFE block (miss and off-road rates) is reported as
descriptive only, as in the other studies.

---

# Fallback: wta_soft (W) vs D

Written and committed **after** the laplace_mixture result (`analyze_seeds_loss.log`: do NOT
adopt; criteria 2, 3 and 4 failed) and **before any wta_soft run**. Tuning split only; the
test split is not used.

## Loss (fixed, unchanged from the section above)

`--loss wta_soft` (`WTASoftLoss`): WTA regression of the closest mode (identical to D's
regression term and its gradient), and the mode scores trained with cross-entropy against
`softmax(-ADE_k / tau)` with the ADE detached, **tau = 1.0 m** (`WTA_SOFT_TAU`, fixed, not a
CLI knob). The model is exactly D's (`train.py` asserts D's parameter count; no scale head),
so with the same `--seed` the initialisation and data order match D's run. Optimiser, LR,
batch size and epochs are unchanged. `verify_loss.py` (CPU, random tensors) must pass; it
runs first in `run_wta_soft_seeds.sh`.

## Runs

| Run | Extra flags | Note |
|---|---|---|
| `D_seed0..4` | none | existing; **not retrained or re-evaluated**. Logs: `eval_D_seed{0,1,2}_e12_rank.log`, `eval_D_seed{3,4}_e12.log` (the same logs used for the laplace_mixture rule) |
| `W_seed0..4` | `--seed s --loss wta_soft` | D's flags, `checkpoint_epoch12.pt`, tuning split, ranking metrics → `eval_W_seed*_e12.log` |

`run_wta_soft_seeds.sh` runs `verify_loss.py`, trains and evaluates W_seed0..4, then runs
`analyze_seeds.py --rule loss --candidate wta_soft --expect-n 2208 --full W_seed0,...,W_seed4
--nomap D_seed0:e12_rank,D_seed1:e12_rank,D_seed2:e12_rank,D_seed3,D_seed4`
→ `analyze_seeds_wta_soft.log`. "full" means W and "no-map" means D.

## Rule

**wta_soft is adopted** only if all four of these hold (seed means over 5 seeds each) —
the same four criteria as for laplace_mixture:

1. W's seed-mean ALL **ADE1** is lower than D's, **and**
2. W's seed-mean **ADE1** is lower than D's in **≥ 6 of the 8** tuning scenes (a tie does
   not count as lower), **and**
3. W's seed-mean ALL **minADE6** is not worse than D's by more than 0.1 m
   (W − D ≤ +0.100 m, using the logged 3-decimal values), **and**
4. W's seed-mean ALL **mode diversity** is **≥ 50%** of D's.

The ADE1 seed std of W and D is reported, not a criterion. If any criterion fails, wta_soft
is not adopted and D stays the baseline.

## Reporting

Reported whatever the outcome: for every run ADE1, minADE6, minFDE6, Brier-minFDE and
diversity, seed mean ± std, and the per-scene ADE1 / minADE6 table. No calibration (W has no
scales). SAFE block descriptive only.
