# Decision rules: map v2 in the Part 1 ego task

Status: pre-registered **before any C_Dmapv2 run was trained** (commit hash in git history). Comparisons are made on the
**tuning split only** first; the test split is used once, through the guard, after the tuning verdicts are
recorded in this file.

## Question

Does the env task's map design (map v2: lane-piece tokens in the agent's own frame, mirror transform) help the
Part 1 ego-trajectory model? The corrected Part 1 rerun found no detectable map effect with D's RGCN lane graph
(map v1). This study changes only the map input and its encoder.

## Setup (fixed)

- Split v2 (`scene_split.py`, `test_scenes_v2.json`), map transform mirror, D's training setup unchanged:
  `--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy --map-transform mirror
  --split-version v2`, batch 8, lr 1e-4, WTA loss, seeds 0, 1, 2. The checkpoint is the final model after
  epoch 12 (`checkpoint_epoch12.pt`). There is no best-epoch selection.
- **Map v2 for the ego** (`--map-version v2`; `config.json` key `map_version`, absent = v1):
  - Tokens: `env_map.LanePolylines` (`lane_polylines_<town>.npz`: CARLA driving lanes plus Town03's Parking
    lanes, taken to V2X-Sim with the mirror). They are placed at the ego's pose (ego_pose translation and
    quaternion yaw) and expressed in the ego frame of `gt_traj`. The parameters are the env task's: every
    lane piece with a point within 30 m, nearest first, at most 48 pieces of at most 11 points. Each point has
    6 features: x/30, y/30, cos and sin of (lane direction − ego heading), is_junction and is_parking.
  - Encoder: `env_model.PolylineEncoder` (per-point MLP, masked max) gives one token per piece. A learned
    "no lane" token is always present. D's `MapCrossAttentionFusion` then runs with the ego embedding as
    query and padded pieces masked. Its output is the map stream: 128 values into D's unchanged 4-stream fuse
    layer.
  - The RGCN lane graph (`topology/*.pkl`) and `nearby_idx` are not used by the v2 model.
  - Everything else (ego, coop, CAM/CPM streams, head, loss) is D's.
- `--map-version v1` (the default) is bit-identical to the code C_D and C_Dnomap were trained at (b25c711).
  `verify_mapv2_off.py` checks this: 50 samples' batches, C_D / C_Dnomap outputs, and initial weights, all
  `torch.equal`, with `RESULT: PASS` committed in `verify_mapv2_off.log`. `verify_mapv2_tokens.py` checks
  the v2 input frame and the model before any training. Its pass bars are: nearest lane median ≤ 1.0 m and
  p95 ≤ 2.5 m on moving train + tuning samples, and median |lane direction − ego heading| ≤ 5°. Its log
  must say `RESULT: PASS`.
- Known limitation: the off-road safety metric uses the same `lane_polylines_<town>.npz`, so map v2 sees the
  lanes the off-road metric scores against. Off-road stays descriptive.
- Known limitation: for fast egos the end of the 6 s trajectory can lie beyond the 30 m lane radius; the same holds
  for fast targets in the env task.

## Runs (`run_mapv2_ego_seeds.sh`; launched by the user)

| name | flags beyond the setup | role |
|---|---|---|
| `C_Dmapv2_seed0..2` | `--map-version v2` | new: D with map v2 |
| `C_Dnomap_seed0..2` | `--no-map` | existing (corrected rerun), reused |
| `C_D_seed0..2` | (none) | existing (map v1), reused, descriptive only |

C_Dnomap is reused, not retrained. It was trained at b25c711 with these flags, and `verify_mapv2_off.py`
shows that the no-map model and its batches are unchanged since. Its tuning logs `eval_C_Dnomap_seed<k>_e12.log`
are reused: `eval_scenes.py`'s tuning path has only gained test-split guard code since b25c711.

Each C_Dmapv2 run is evaluated (`eval_scenes.py`, tuning split, epoch 12, `eval_C_Dmapv2_seed<k>_e12.log`)
after all three are trained. A run that crashes is re-run from scratch with the same seed and flags. No run is
repeated for any other reason.

## Primary rule (`analyze_seeds.py --rule map`; Part 1's map rule)

With seed means and sample standard deviations (n = 3) of minADE6 (best-of-6 ADE, tuning split):
map v2 helps only if C_Dmapv2's seed-mean minADE6 is lower than C_Dnomap's in **≥ 7 of the 8 tuning scenes**
AND the ALL gap (C_Dnomap − C_Dmapv2) is **> 2 × the larger of the two seed standard deviations**.
Otherwise: no detectable effect.

## Secondary rule (`analyze_conditions.py --rule corrected --map-full Dmapv2`)

**Map v2 lowers minADE6 on turning vehicles.** The turning label and the pooled form are those of
runs/DECISION_RULES_CORRECTED.md. Scene 30 has no turning sample (v2 tuning: Town03 80, Town04 0,
Town05 64), so no per-scene count applies. Map v2 helps on turning only if C_Dmapv2's seed-mean turning
minADE6 is below C_Dnomap's AND (C_Dnomap − C_Dmapv2) > 2 × max(seed std C_Dmapv2, seed std C_Dnomap),
on the pooled tuning split (`COND ALL turning`).

## Reported descriptively (no verdict)

- ADE1, minFDE6, per-scene and per-town rows; other conditions; miss and off-road rates.
- C_Dmapv2 vs C_D (map v1): ALL rows of both, from the existing C_D logs.
- minADE6 and ADE1 on MOVING ego samples, C_Dmapv2 vs C_Dnomap (seed mean +/- std, the `moving` condition rows of
  `analyze_conditions_mapv2_ego.log`), to set next to the env task's moving-target map hypothesis. No verdict.

## Commands (run by `run_mapv2_ego_seeds.sh` after the evaluations)

```
python analyze_seeds.py --rule map --full C_Dmapv2_seed0,C_Dmapv2_seed1,C_Dmapv2_seed2 \
    --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2               > analyze_seeds_mapv2_ego.log
python analyze_conditions.py --rule corrected --map-full Dmapv2       > analyze_conditions_mapv2_ego.log
```

## Test evaluation

The test runs only after both tuning verdicts are written into this file's Results section. Each
C_Dmapv2_seed<k> epoch-12 checkpoint is evaluated once on the split v2 test split through the guard
(`eval_scenes.py --split test --final-test`, `final_test_guard.py`; `eval_C_Dmapv2_seed<k>_test_final.log`,
recorded in `runs/TEST_EVALS.log` before it evaluates; a re-run only after a crash with no result line).
C_Dnomap is not re-evaluated: its final-test logs `eval_C_Dnomap_seed<k>_test_final.log`
(runs/DECISION_RULES_FINAL_TEST.md, f322abb) are reused. The same two rules are then applied to the test
logs (`--split test --suffix test_final`; "8 tuning scenes" reads "8 test scenes"). No rule changes after the
tuning results. The test needs its own run script, prepared after the tuning verdicts.

## Results
(empty until the runs finish)

### Tuning verdicts (15 Oct 2026; analyze_seeds_mapv2_ego.log, analyze_conditions_mapv2_ego.log)
- Primary (ALL minADE6): C_Dmapv2 2.291 +/- 0.192 vs C_Dnomap 2.277 +/- 0.220, better in 5/8 scenes,
  gap -0.014 (bar 0.439). VERDICT primary: no detectable map effect.
- Secondary (turning, pooled, n=144): 10.475 +/- 0.463 vs 9.999 +/- 0.923, gap -0.476 (bar 1.845).
  VERDICT secondary: no detectable effect.
- Descriptive: ADE1 5.531 vs 5.218 (no-map); map v1 (C_D) for reference minADE6 2.531, ADE1 6.772.

### Final test verdicts (17 Oct 2026; run_mapv2_ego_final_test.sh, 3 guarded evaluations)
- Primary (ALL minADE6): C_Dmapv2 2.725 +/- 0.148 vs C_Dnomap 2.708 +/- 0.259, better in 4/8 scenes,
  gap -0.017 (bar 0.518). VERDICT primary (test): no detectable map effect.
- Secondary (turning, pooled, n=336): 7.155 +/- 1.284 vs 6.801 +/- 1.372, gap -0.355 (bar 2.744).
  VERDICT secondary (test): no detectable effect.
- Descriptive: ALL ADE1 5.668 +/- 0.495 vs 5.909 +/- 0.300 (better in 6/8); map v1 (C_D) test ADE1 6.749.
  Moving minADE6 4.328 +/- 0.294 vs 4.249 +/- 0.492 (n=1116). Test Town04: 17 moving, 0 turning samples.
