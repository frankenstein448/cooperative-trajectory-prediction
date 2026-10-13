# Decision rules: Part 2, environment prediction (EgoOnly-env, V2X-env, V2X-env-nomap)

Written and committed **before any env-task model was trained or evaluated**. Comparisons are made on the
**tuning split v2 only**; the test split is not used by this document.

## Setup (fixed)

- Task: predict the future (30 steps = 6 s at 5 Hz) of the other vehicles around the ego (`train.py --task env`;
  `runs/DESIGN_ENV.md` sections 9–10). Targets: vehicles within 50 m of the ego with a full 8-step history and
  30-step future (`build_env_target_cache.py`, cache commit af265e9). Visibility (num_lidar_pts ≥ 5):
  (a) the ego sees the target; (b) not (a), but a coop CAV 1–4 within 200 m sees it; (c) neither.
- Option B1: a model knows only the targets it can observe. EgoOnly-env: class (a). V2X-env and
  V2X-env-nomap: (a) + (b), plus the cooperative sensor stream. Class (c) is never an input or a target; it
  counts as uncovered.
- Common sample list: 13,384 samples (D's samples with at least one class-(a) target), split v2
  (`scene_split.py`; tuning scenes Town03 85, Town04 30, Town05 3, 14, 15, 16, 21, 22).
- Map: map transform mirror for positions and headings. Map v2 tokens come from `lane_polylines_<town>.npz`
  (Driving + Town03 Parking, lane type as a feature), in the target's own frame. The ego CAM keeps only the
  speed (x, y and world heading dropped).
- Training: 12 epochs, seeds 0, 1, 2, batch 8, lr 1e-4, D's WTA loss on the known targets, evaluation of
  `checkpoint_epoch12.pt`. Flags: `--task env --epochs 12 --use-image-cache --use-speed --normalize-cam
  --drop-cam-xy --map-transform mirror --split-version v2`, plus:

| run | flags | known targets |
|---|---|---|
| `E_EgoOnly_seed0..2` | `--v2x off` | (a) |
| `E_V2X_seed0..2` | `--v2x on` | (a) + (b) |
| `E_V2Xnomap_seed0..2` | `--v2x on --no-map` | (a) + (b) |

## Metrics and labels (`env_metrics.py`, `eval_env.py`)

Per target, own frame: minADE6 (best of 6 modes), ADE1 (top-scored mode), minFDE6, FDE1, miss rates
(FDE > 2 m, top-1 and best-of-6); constant velocity (the frame-1 → frame displacement repeated) and stay-put
baselines (ADE, FDE, miss). Labels, from ground truth only, Part 1's definitions: **moving** = nonzero
frame-1 → frame displacement; stationary = not moving; **turning** = moving and |heading change over the 30-step
GT| > 30° (`scene_conditions.heading_change_deg`); straight = moving and not turning. Category: car (car, police)
vs two-wheeler (bicycle, motorcycle).

## Rule form (`env_metrics.rule`, `analyze_env.py`)

All rules use **MOVING targets** and **minADE6**, seed means and sample standard deviations (n = 3), on the
8 tuning scenes:

- **Full form** (every tuning scene has at least one target of the rule's subset): "with" is better only if its
  seed-mean is lower in **≥ 7 of the 8 tuning scenes** AND the ALL gap (without − with) is **> 2 × the larger
  of the two seed standard deviations**.
- **Pooled form** (at least one tuning scene has no target of the subset): "with" is better only if its
  seed-mean ALL is lower AND the same gap condition holds.

Which form applies depends only on the ground-truth labels, which are identical for every run.
`analyze_env.py` prints the per-scene counts and the form used. A deterministic baseline enters with seed
standard deviation 0.

## Pre-registered comparisons

1. **V2X on (a):** E_V2X vs E_EgoOnly, minADE6 on moving class-(a) targets (the targets both models predict).
2. **V2X on (b):** E_V2X's minADE6 vs the constant-velocity baseline's ADE, on moving class-(b) targets. EgoOnly
   cannot predict (b) at all. Note that this compares the best of 6 modes with a single prediction, which favours
   the model. E_V2X's ADE1 against the same baseline is reported next to it, descriptively.
3. **Map:** E_V2X vs E_V2Xnomap, minADE6 on moving targets ((a) + (b)). **Secondary:** the same on turning
   targets.

No other comparison gets a verdict, and no correction for multiple comparisons is applied.

## Reported descriptively

- **Coverage per model:** predicted targets / all kept targets (a + b + c) over every D sample of the tuning
  scenes, also with the window-filtered vehicles (`n_removed`) in the denominator.
- ADE1, minFDE6, FDE1 and miss rates; all targets, stationary, straight; (a) / (b); car / two-wheeler; per scene
  and ALL; the constant-velocity and stay-put baselines for every row.

## Procedure

`run_env_seeds.sh` (launched by the user) refuses to start unless:
- the code, this document, the split and the lane polylines are committed;
- the env cache's manifest says `builder_dirty=False`;
- `verify_env_model.log` (the real-data check of Stage 2) is committed with `RESULT: PASS`;
- no `runs/E_*` directory exists.

Order: E_EgoOnly ×3, E_V2Xnomap ×3, E_V2X ×3. Each config's three runs are evaluated (`eval_env.py`, epoch 12,
tuning) right after training; `analyze_env.py` runs at the end. A run that crashes is re-run from scratch with
the same seed and flags; no run is repeated for any other reason. All verdicts are reported whatever they say.
