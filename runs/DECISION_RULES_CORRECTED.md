# Decision rules: corrected Part 1 rerun (split v2, map transform mirror)

Written and committed **before any run of this study was trained**. Comparisons are made on the
**tuning split only**; the test split is not used by this document.

## Why this rerun

Two errors were found on 2026-10-13. **Every earlier Part 1 map result is superseded by this study**,
because each one used both:

1. **Legacy map transform.** `coord_transform.TOWN_TRANSFORMS` (per-town rotations) is wrong away from where
   it was fitted. No scene passes median ≤ 1.0 m AND p95 ≤ 2.5 m (moving vehicles to the nearest lane
   waypoint): scene medians are 3.3–24.9 m (`check_map_alignment.log`). The correct relation is a
   mirror: CARLA = (x, −y) of V2X-Sim, with yaw_carla = −yaw_v2x. Under it all 54 scenes pass (lanes with
   Town03's Parking lanes; medians 0.54–0.66 m, p95 1.24–1.78 m). The median angle between each vehicle's
   heading and its nearest lane's yaw is 0.03–0.11° (`check_map_alignment_mirror.log`,
   `verify_map_transform.log`). D's map stream had therefore been looking up the wrong lanes.
2. **Leaky v1 split.** Scenes cut from one SUMO simulation overlap in time. Scene 49 (test) equals scene 72
   (train), and 35 scene pairs in different splits shared moving-vehicle positions (`check_scene_overlap.log`).

Superseded: the D vs Dnomap map rule, the EgoOnly / NoCPM / NoCoop component results, the condition
results, and the confirmatory turning test on the v1 test split. They stay in the repository as the record
of what was run.

## Setup (fixed)

- **Split v2** (`scene_split.py`, `test_scenes_v2.json`; approved, committed before any v2 run). Whole
  overlap components (scenes sharing > 1% of moving-vehicle positions) per split, stratified by town:
  train 10,488 / tuning 2,208 / test 2,208 samples. Tuning scenes: Town03 85; Town04 30;
  Town05 3, 14, 15, 16, 21, 22. No pair of scenes in different splits shares > 1% of moving positions
  (`check_scene_overlap_v2.log`: 0 pairs). Test scene 78 was evaluated once in Part 1 (v1 test split; no model
  choice depended on it). No v1 tuning scene is in the v2 test split.
- **Map transform mirror** for the map lookup, the off-road metric and the junction distance
  (`coord_transform.py`). D's RGCN lane graph (`topology/*.pkl`) is **unchanged**, so this rerun isolates
  the alignment fix. The off-road metric uses all exported lanes (`lane_polylines_<town>.npz`: driving lanes,
  plus Town03's Parking lanes). GT off-road control on v2 tuning: Town03 0.00%, Town04 0.00%, Town05 3.51%,
  none flagged UNRELIABLE. Under legacy it was 32% / 29% / 47% (`check_offroad_gt.log`).
- D's training setup otherwise unchanged: `--epochs 12 --use-image-cache --use-speed --normalize-cam
  --drop-cam-xy`, batch 8, lr 1e-4, WTA loss, seeds 0, 1, 2, evaluation on `checkpoint_epoch12.pt`.
  Both new options are passed explicitly: `--map-transform mirror --split-version v2`. Both are recorded in
  each run's `config.json`.

## Runs (`run_corrected_seeds.sh`; launched by the user)

| name | flags beyond the setup | role |
|---|---|---|
| `C_Dnomap_seed0..2` | `--no-map` | D without the map |
| `C_EgoOnly_seed0..2` | `--no-cpm --no-coop` | D without CPM objects and cooperative sensors |
| `C_D_seed0..2` | (none) | full model |

Order: C_Dnomap, then C_EgoOnly (the faster ones), then C_D. Each config's three runs are evaluated
(`eval_scenes.py`, tuning split, epoch 12, `eval_<run>_e12.log`) right after training. The analyses below
run once C_D is evaluated. A run that crashes is re-run from scratch with the same seed and flags; no run is
repeated for any other reason.

## Primary rules (`analyze_seeds.py`; the same rule as Part 1)

With seed means and sample standard deviations (n = 3) of minADE6 (best-of-6 ADE, tuning split):

1. **Map** (`--rule map`, full = C_D, no-map = C_Dnomap): the map helps only if C_D's seed-mean minADE6 is
   lower than C_Dnomap's in **≥ 7 of the 8 tuning scenes** AND the ALL gap (C_Dnomap − C_D) is
   **> 2 × the larger of the two seed standard deviations**.
2. **V2X** (`--rule component`, full = C_D, ablation = C_EgoOnly): the same rule. CPM + cooperative sensors
   help only if C_D is lower in ≥ 7 of 8 tuning scenes AND (C_EgoOnly − C_D) > 2 × the larger seed std.

## Secondary pre-registered hypothesis (`analyze_conditions.py --rule corrected`)

**The map lowers minADE6 on turning vehicles.** The turning label is unchanged (`scene_conditions.py`:
moving, and |heading change over the 30-step GT| > 30°). The per-scene part of the rule cannot be applied to
turning: one tuning scene (Town04's scene 30) has **no** turning sample (v2 tuning: Town03 80, Town04 0,
Town05 64; 144 in all). These counts come from ground-truth labels, read before any model of this study
existed. So the rule is the pooled one Part 1 used for conditions
(`DECISION_RULES_CONDITIONS.md` / `DECISION_RULES_TURNING_TEST.md`): the map helps on turning vehicles
only if C_D's seed-mean turning minADE6 < C_Dnomap's AND (C_Dnomap − C_D) > 2 × max(seed std C_D,
seed std C_Dnomap), on the pooled tuning split (`COND ALL turning` rows).

## Reported descriptively (no verdict)

- ADE1, minFDE6, per-scene and per-town rows, for every run.
- Conditions: stationary, moving, straight, near_junction, not_near_junction for map and V2X; turning for V2X.
- Safety: miss rates (top-1, best-of-6) and off-road rates (top-1, all modes), with the GT off-road control
  per town.

## Commands (run by `run_corrected_seeds.sh` after C_D's evaluations)

```
python analyze_seeds.py --rule map --full C_D_seed0,C_D_seed1,C_D_seed2 \
    --nomap C_Dnomap_seed0,C_Dnomap_seed1,C_Dnomap_seed2                  > analyze_seeds_corrected_map.log
python analyze_seeds.py --rule component --full C_D_seed0,C_D_seed1,C_D_seed2 \
    --nomap C_EgoOnly_seed0,C_EgoOnly_seed1,C_EgoOnly_seed2               > analyze_seeds_corrected_v2x.log
python analyze_conditions.py --rule corrected                           > analyze_conditions_corrected.log
```

All three verdicts are reported whatever they say, with per-run and per-scene numbers. The test split v2
is not evaluated under this document; any test evaluation needs its own committed rules first.
