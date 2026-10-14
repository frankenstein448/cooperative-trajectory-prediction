# Part 2 design: environment prediction (Stage 2, for review)

Status: design accepted with the decisions in section 9; no model code yet. Stage 1 cache: `~/hd_cache/env_targets/` (`build_env_target_cache.py`,
verified by `verify_env_targets.py`). V2X-Sim is 5 Hz (arXiv 2202.08449), so the horizon is 30 frames = 6 s
and the history is 8 frames = 1.6 s.

## 0. Decision recorded for Stage 4 (fixed now)

The pre-registered rules (V2X on (a), V2X on (b), map) are applied to **moving targets only**: cache
`speed > 0`, i.e. a nonzero frame-1 -> frame displacement (the Part 1 "stationary" definition applied to the
target). Results on ALL targets and on stationary targets are reported descriptively and decide nothing.
Tuning-split sizes: (a) 35,608 targets, 16,228 moving; (b) 13,160 targets, 6,192 moving.
Overall, 43.6% of kept targets are moving, about the same in (a), (b) and (c).

## 1. Blocker for the map component: map alignment

The map tokens need the target's position in CARLA coordinates (`coord_transform.v2xsim_to_carla`).
I checked it on the cache: distance from each **moving** target to the nearest `dense_waypoints` point.
Lane centres are 2 m apart along a lane, so a vehicle on a lane should be within about 2 m.

| town | moving CAVs: median / >5 m | moving non-CAVs: median / >5 m |
|---|---|---|
| Town03 | 1.5 m / 27% | 2.3 m / 34% |
| Town04 | 1.7 m / 33% | 4.1 m / 46% |
| Town05 | 1.6 m / 27% | 1.7 m / 28% |

The error depends on the scene. Scenes 37, 38 and 42 have a median of 1.0 m and essentially nothing beyond 5 m.
Scenes 6, 9, 20 (Town04), 14 and 21 (Town05) have a median of 9–15 m and more than 60% beyond 5 m. Even the CAVs,
which drive on lanes, are often far from any waypoint. So the per-town transform does not fit every scene,
or some scenes are not in the town assigned to them, or the waypoints miss some roads.
**D's nearest-lane lookup uses the same transform, so this may also affect the Part 1 map results.**

Proposal: before any map code, add a read-only diagnostic, `check_map_alignment.py`. It reports, for each scene,
the distance of moving CAVs to the nearest waypoint, a per-scene best-fit offset/rotation, and the town that fits
best. Then you decide the fix (per-scene transforms, corrected scene->town mapping, or excluding scenes from the map
comparison). The map comparison (V2X-env vs V2X-env-nomap) should not be pre-registered until a fixed
threshold passes, for example the median distance of moving CAVs to the nearest waypoint <= 2 m in every used
scene.

## 2. Map tokens ("map v2"): the data needs lane structure

`dense_waypoints_<town>.npy` contains only unordered (x, y) points: no lane id, no order, no direction. So
polylines cannot be read from it directly. Options:

- **M1 (recommended): one-time CARLA export**, `export_lane_polylines.py` (carla-venv, CARLA running with
  the town loaded, the same way `dense_waypoints.py` worked). It uses the same `generate_waypoints(2.0)` call,
  so the point set should match `dense_waypoints` exactly (to be checked), plus `road_id`, `section_id`, `lane_id`,
  `s`, `is_junction` and yaw. Points are grouped by (road, section, lane), ordered by `s` and cut into
  pieces of 10 points (about 20 m). Junction lanes are included. Output: `lane_polylines_<town>.npz`.
- M2: chain the unordered points by nearest neighbour. With lanes 3.5 m apart and points 2 m apart this is
  ambiguous at junctions and between parallel lanes. Not recommended.
- M3: one token per waypoint, with no polylines and no direction. Simplest, but the direction of travel is lost.

Per-target tokens (M1): every polyline piece with any point within 30 m of the target, nearest first, capped at
K = 48 (about 200 waypoints are within 30 m, p95 about 320, so roughly 20–35 pieces). Point features are
in the **target's frame** (x, y relative to the target, rotated by its yaw), plus the segment direction
(cos, sin) in that frame and an `is_junction` bit. No absolute coordinates are used. Tokens are built on the fly
in the dataset with one KD-tree per town over piece points (cheap). Encoder: a PointNet-style polyline encoder,
MLP(5 -> 64 -> 128) per point, then max over the piece's points, giving one 128-d token per piece, with padding
masks.

## 3. Data loading: `env_dataset.py`

- **Samples:** D's sample list (PilotDataset with `require_prev_frame=True`, `use_speed`, image cache,
  `normalize_cam`, `drop_cam_xy`, as D), minus samples with no class-(a) target. That drops 1,520 samples:
  292 at frames 6–7, where no target can have 8 frames of history, and the rest frames with no visible
  vehicle. Every sample with a (b) target also has an (a) target, so **all three models use the same
  sample list**. The cache is joined by sample index; init asserts that (agent, scene, frame) match row for row.
- **CPM off** in the env task (`use_cpm=False`). D's CPM holds every GT box within 50 m, including class (c),
  so it would leak uncovered targets. The ego's own CAM is kept.
- **Coop sensors** are loaded only with `--v2x on` (`use_coop`), exactly as D loads them.
- **Known targets per sample:** `--v2x off`: vis == 0 (mean 16.0, p95 28, max 39). `--v2x on`: vis <= 1
  (mean 21.3, p95 37, max 45). Class (c) is never loaded as input or target.
- **Per-target inputs:** history in the own frame [8, 2], normalized with train-split stats (like
  `history_norm`, saved to `env_norm.json` in the run dir). State: speed, size (w, l), category one-hot (4),
  and the pose relative to the ego (x/50, y/50, cos, sin of `yaw_ego`). Targets: `fut_own` [30, 2] and
  `fut_mask`.
- **Assumption, consistent with Part 1's CPM:** states and histories of known targets come from GT
  annotations, i.e. perfect tracking. History frames from before the target became visible are included.
- **Collate:** pad to the largest target count in the batch: `[B, Tmax, ...]` plus `target_mask`;
  polylines `[B, Tmax, K, 10, 5]` plus masks.

## 4. Model: `env_model.py`, `EnvFusion`

d = 128 throughout, 4 heads, pre-LayerNorm, dropout 0.1.

1. **Scene tokens:** D's `EgoStream` (ego LiDAR + 3 cameras) gives 1 token; D's `CAMEncoder` on the ego's
   CAM gives 1 token. With `--v2x on`, D's `CoopSensorStream` adds 1 token (mean over coop CAVs; a zero vector
   plus a "no coop" flag when there are none). Each token gets a learned type embedding. These are D's
   encoders unchanged, and each gives a single pooled vector, so they carry scene context but no
   per-target spatial detail. Spatial (sector/BEV) tokens would be an architecture change; I don't propose
   it for this round.
2. **Target query:** GRU history encoder (`history_encoder.HistoryEncoder`, D's GRU) plus MLP(state), summed.
3. **Two blocks, each:**
   - (a) **Interaction:** attention over the other known targets plus the ego, which is itself an agent with its
     CAM token at the ego origin. The relative pose of j in i's frame (dx, dy, cos, sin of dyaw), through an
     MLP, is added to j's key and value, so the interaction is agent-centric. Pairs: B x T x T, at most
     8 x 46 x 46, which is cheap.
   - (b) **Map:** attention over the target's own polyline tokens (skipped with `--no-map`).
   - (c) **Scene:** attention over the scene tokens.
   - (d) FFN.
4. **Head:** D's `TrajectoryHead(128, 6 modes, 30)` per target, giving 6 x 30 x 2 in the target's frame plus
   6 logits.

## 5. Loss

D's `WinnerTakesAllLoss` (smooth-L1 on the winning mode, CE on the logits, weight 1), applied to the known
targets of the batch flattened to `[N, 6, 30, 2]`. Each target has equal weight. Padding is excluded and all
30 steps are valid by construction. No reweighting of moving vs stationary targets in training (the rule
in section 0 only concerns evaluation).
Model selection: the epoch-12 checkpoint, as in Part 1.

## 6. Flags (`train.py`)

- `--task {ego,env}`, default `ego`. The `ego` path is untouched: same code, same `EndToEndFusion`.
- `--v2x {off,on}` (required with `--task env`). `--no-map` removes the polyline encoder and the map
  attention (removal, not zeroing, as D's `--no-map`).
- With `--task env`, these are rejected: `--no-cpm`/`--no-coop` (replaced by `--v2x`), `--history-frames`,
  `--loss` other than `wta`, and `--degrade` (not in scope).
- Run names: `EgoOnlyEnv_seed{0,1,2}` (`--v2x off`), `V2XEnv_seed*` (`--v2x on`), `V2XEnvNomap_seed*`
  (`--v2x on --no-map`).
- **Old-task identity check:** `--task ego` gives the same parameter count as D, loads
  `runs/D_seed0/checkpoint_epoch12.pt` with strict=True, and gives the same first-batch loss as before the
  change.

## 7. Compute and memory (per epoch, RTX 5090)

Measured in Part 1: D (ego + coop + map) 247 s/epoch; EgoOnly (no coop) 89 s/epoch, batch 8. The sensor
encoders dominate. The per-target part is small: about 8 x 21 targets x 35 pieces x 10 points through
a small MLP, plus attention over at most 46 x 46 pairs. Expected, to be measured in the dry run:

| model | samples (train) | est. s/epoch | 12 epochs |
|---|---|---|---|
| EgoOnly-env | ~9,400 | 85–110 | ~20 min |
| V2X-env | ~9,400 | 220–270 | ~50 min |
| V2X-env-nomap | ~9,400 | 210–260 | ~50 min |

That is roughly 6 hours for 3 x 3 runs, plus evaluations. GPU: well under 32 GB (polyline activations about
0.1 GB per layer per batch). RAM: the dataset keeps only the numeric arrays it needs from the cache
(about 150 MB), shared copy-on-write with the 2 workers. Polylines are a few MB per town. Map tokens built on
the fly: about 1–2 ms per sample, so loading is not the bottleneck.

## 8. Verification plan for the Stage 2 code (`verify_env_model.py`, about 50 samples)

1. Dataset vs cache: target sets per model, own-frame arrays, and that no (c) target ever appears; with
   `--v2x off`, no (b) target and no coop sensors.
2. Map tokens: no absolute coordinates; a target's own lane piece passes within about 1 m of the origin with
   direction about 0 deg, on the aligned scenes. **Changed after the first result (user, 2026-10-14):** the implemented bar
   ">= 90% of moving targets within 1.0 m" gave 0.852 (p50 0.10 m, p90 1.30 m, direction 0.08 deg), FAIL. It had
   been set without reference to real driving; the misses are the expected tail (lane changes, junction turns,
   off-centre driving; median lateral offset ~0 in every town, not a misalignment). Replaced by: median distance
   <= 0.5 m AND median direction error <= 5 deg AND >= 95% of moving targets within 2.5 m (the alignment
   acceptance's p95 tolerance, section 9).
3. Invariance: rotating and translating the world leaves the per-target inputs and outputs unchanged.
4. Loss: equals D's `WinnerTakesAllLoss` on the same flattened tensors; padding has zero gradient.
5. `--task ego` identity (section 6), then a one-epoch dry run per model with the measured s/epoch.

## 9. Decisions (user, 2026-10-13)

1. **Map alignment first**, before any Stage 2 code: `check_map_alignment.py` (read-only) with a per-scene
   refit of (theta, tx, ty) that minimises the robust distance of all moving vehicle positions (CAVs included) to
   the lane waypoints (grid search as in the existing transform fitting, then refine). **Acceptance threshold,
   fixed now: per scene, median <= 1.0 m AND p95 <= 2.5 m.** Pass/fail is reported per scene for the old
   per-town transform and the new per-scene one; scenes failing both are listed for exclusion. Part 1 impact:
   which tuning/test scenes were misaligned under the old transform. Nothing that existing models use
   (`coord_transform.py` etc.) changes before a report.
2. **Map source: M1**, a CARLA export with lane structure, one town per CARLA start, then `pkill CarlaUE4`.
   The exported points must reproduce `dense_waypoints_<town>.npy` exactly.
3. **Scene tokens:** D's pooled encoders this round.
4. **CPM dropped in the env task; common sample list of 13,384 samples for all three models.**

Order: alignment check -> report -> CARLA export -> report. No Stage 2 model code yet.

## 10. Stage 2 implementation (2026-10-13), updated with today's decisions

Code: `env_map.py` (map v2 tokens), `env_dataset.py` (EnvDataset, pure feature functions, padding collate),
`env_model.py` (EnvFusion, env_wta_loss), `train.py --task env --v2x {off,on} [--no-map]`. Tests:
`test_env_stage2.py` (synthetic tensors only; no metadata, PilotDataset, cache or GPU).

Changes against sections 2–5, all following the decisions made since:
- **Map transform mirror for positions and headings.** Lane points and directions are taken from CARLA back to
  V2X-Sim (`carla_to_v2xsim_batch`, `carla_yaw_to_v2xsim`), then expressed in the target's own frame (box
  yaw). That is the frame of `hist_own` / `fut_own`, so map tokens and trajectories share one frame. No
  absolute coordinates.
- **Map v2 tokens** from `lane_polylines_<town>.npz` (Driving + Town03 Parking). Each token is one lane piece
  (≤ 11 points in travel order). Every piece with a point within 30 m of the target, nearest first, at most 48.
  Point features (6): x/30, y/30, cos and sin of (lane direction − target heading), is_junction,
  **is_parking** (lane type). A learned "no lane" token is always present, so a target with no piece in range is
  still well defined.
- **Split v2:** `three_way_scene_split(dataset, split_version=...)` on `EnvDataset.samples`. The cache's own
  `sample_split` field is the v1 split and is not used. The cache does not need rebuilding.
- **Samples:** 13,384 (D's list minus samples without a class-(a) target), the same for all models; EnvDataset
  asserts the count and that the cache rows match PilotDataset's sample list exactly.
- **Inputs per target:** history (own frame, z-scored with train-split `env_norm.json`), state (pose relative
  to the ego /50, cos and sin yaw, speed / 5, size w and l / 5, category one-hot), and the ego-frame pose for
  the relative-pose interaction.
- **Ego CAM, speed only (decision 2026-10-13, before any env result):** after normalization, `env_cam()`
  zeroes x, y (as D's drop_cam_xy) **and the ego's absolute world heading**; only the speed is kept.
  `train.py --task env` requires `--use-speed --normalize-cam --drop-cam-xy`, and `config.json` records
  `env_cam`. Test: `test_env_stage2.test_env_cam_keeps_only_speed`.
- **Coop with no CAV in range:** a learned "no coop" token. D's CoopSensorStream returns a CPU zero vector in
  that case.
- **Parameters:** EgoOnly-env 896,494; V2X-env 1,061,262; V2X-env-nomap 903,246 (D: 740,942).
- **Ego task unchanged:** `train.py`'s epoch loop now takes a step function. Its default (`ego_step`) gives
  the same losses and bit-identical parameters as the pre-env `run_epoch`
  (`test_env_stage2.test_ego_run_epoch_unchanged`). The ego dry run additionally prints the per-batch time and
  the peak GPU memory.

Still to do once the GPU is free (needs the data):
1. `verify_env_model.py` on about 50 real samples: EnvDataset vs the cache (target sets per model, no (c)
   target, no coop with `--v2x off`); map tokens on real scenes (the target's own lane piece passes within about
   1 m of the origin at about 0°); `--task ego` still strict-loads D's checkpoint with D's parameter count.
2. Dry runs of the three env configs (per-batch time, peak memory).
3. Stage 3 (evaluation) and Stage 4 (rules and run scripts), as planned.

## 11. Stages 3 and 4 (2026-10-13; code and synthetic tests only)

- `env_metrics.py`: per-target minADE6 / ADE1 / minFDE6 / FDE1 / miss rates (2 m), constant-velocity and
  stay-put baselines, condition labels (Part 1 definitions on the own-frame ground truth), subsets
  ((a)/(b), moving/stationary/turning/straight, car/two-wheeler), coverage, and the rule computation
  (`rule`: full form ≥ 7/8 scenes AND gap > 2 × larger seed std; pooled form when a scene has no target of the
  subset).
- `eval_env.py`: tuning split only (the test split is refused); per tuning scene and ALL; writes
  `runs/<run>/eval_env_tuning_<ckpt>.npz` with per-target arrays (cache target id, scene, labels, metrics,
  baselines) and coverage. EnvDataset gives two extra non-input fields: `t_tid` and `t_cv_vel`.
- `analyze_env.py`: checks the runs (split, scenes, transform, identical targets across seeds, EgoOnly's targets
  = V2X's (a) targets), then applies `runs/DECISION_RULES_ENV.md`.
- `run_env_seeds.sh`: preflight as in `run_corrected_seeds.sh`, plus the cache manifest and a committed
  `verify_env_model.log` with `RESULT: PASS`.
- Tests: `test_env_stage2.py`, 21/21 on synthetic data (hand-computed metric values, labels, subsets, coverage,
  rule forms, `collect` on a padded batch, `analyze_env.py` end to end on fake per-run npz files).
