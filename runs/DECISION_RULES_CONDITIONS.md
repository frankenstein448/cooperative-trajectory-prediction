# Decision rules: where do the map and V2X help? (condition-based evaluation)

Written and committed **before any condition-based number was computed for any model**. Evaluation
only: no training. The existing epoch-12 checkpoints are re-evaluated on the **tuning split only**;
the test split is not used.

## Runs (existing checkpoints, `checkpoint_epoch12.pt`, unchanged)

- `D_seed0..2`: full model (map + V2X)
- `Dnomap_seed0..2`: D without the map
- `EgoOnly_seed0..2`: D without CPM objects and cooperative sensors (`--no-cpm --no-coop`)

`run_condition_evals.sh` re-runs `eval_scenes.py` for all nine and writes `eval_<run>_e12_cond.log`.
Every re-run's accuracy rows must match its original `eval_<run>_e12.log` **exactly** (same checkpoint
and data, so the numbers must match). Any difference stops the script before the analysis.

## Conditions (from ground truth only, `scene_conditions.py`; the same labels for every model)

Every tuning sample (2,208) is labelled:

| Condition | Definition |
|---|---|
| `stationary` | frame-1 → frame displacement = 0 (the existing definition, `motion_utils.get_cv_velocity`) |
| `moving` | frame-1 → frame displacement > 0 (reported, not in the rule) |
| `turning` | moving **and** \|heading change over the 30-step GT\| > 30° |
| `straight` | moving **and not** turning |
| `near_junction` | current position within 20 m of a junction area (below). Independent of motion |
| `not_near_junction` | the complement (reported, not in the rule) |

**Heading change.** The GT path is the current position (origin of the agent-centric GT) followed by
the valid GT points in order. Its steps are the displacements between successive points. Steps shorter
than 0.5 m are ignored. Heading change = direction of the last remaining step − direction of the first,
wrapped to (−180°, 180°]. If fewer than 2 steps remain (a slow mover, < 2.5 m/s at 5 Hz), the change is
0, so the sample counts as `straight`. The number of such samples is printed (`COND note`).

**Junction areas.** These come from CARLA 0.9.15's OpenDRIVE files
(`~/carla_sim/CarlaUE4/Content/Carla/Maps/OpenDrive/Town0X.xodr`), loaded offline as
`carla.Map(name, xodr)` by `export_junctions.py`. That script first checks that this offline map's
`generate_waypoints(2.0)` reproduces `dense_waypoints_town0X.npy` (from the live server) exactly. It
then saves every lane-centre waypoint with `is_junction = True` (on a connecting road inside an
OpenDRIVE `<junction>`), at 1 m spacing along every lane, as `junction_waypoints_town0X.npy` (Town03:
31 junctions / 6,156 points; Town04: 27 / 8,496; Town05: 21 / 6,646). An agent is `near_junction` if
its current position, taken into CARLA coordinates with its raw pose and `coord_transform` (as in the
off-road metric), is ≤ 20 m from the nearest of these points. The distance is to junction lane
centres, not to the junction's paved edge, which is about half a lane width (~1.75 m) closer.

Before any model was evaluated, the labels were counted on the ground truth of the 8 tuning scenes
(all agent frames, not just the evaluated samples): most frames are within 20 m of a junction (median
distance 4.5–21 m per scene). `near_junction` will therefore cover most of the tuning set and stay close
to ALL. Under 30° turns are rare (roughly 10–20% of moving frames). Many moving frames in some scenes
are slow (< 2 steps ≥ 0.5 m) and count as straight. The definitions were not changed after these counts.

## Metric

minADE6 (best-of-6 ADE) per condition, over all 8 tuning scenes pooled (the `COND ALL <condition>`
row). ADE1 and minFDE6 are reported and are not part of the rule. Per-town rows are descriptive.

## Rule (applied separately to each condition: turning, straight, stationary, near_junction)

With seed means *m* and sample standard deviations *s* (n = 3 seeds) of that condition's minADE6:

- **Map helps on the condition** only if *m*(D) < *m*(Dnomap) **and**
  *m*(Dnomap) − *m*(D) > 2 × max(*s*(D), *s*(Dnomap)).
- **V2X helps on the condition** only if *m*(D) < *m*(EgoOnly) **and**
  *m*(EgoOnly) − *m*(D) > 2 × max(*s*(D), *s*(EgoOnly)).

Otherwise the result for that condition is "no detectable effect". Implemented in
`analyze_conditions.py`. No correction for testing 4 conditions × 2 comparisons is applied. This is
noted when the results are reported.

## Sanity checks (in `analyze_conditions.py`; any failure stops it)

- The ALL row has n = 2,208 in every log.
- Every condition's n is identical across all nine runs (labels depend on ground truth only).
- stationary + moving = ALL, turning + straight = moving, near + not-near = ALL.
- COND stationary n equals the existing `stationary (ALL)` n in the same log.

## Reporting

All four conditions and both comparisons are reported whatever the outcome, with per-run values,
seed mean ± std, ADE1 and minFDE6, and per-town rows.
