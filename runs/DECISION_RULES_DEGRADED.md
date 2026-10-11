# Decision rules: does the map help when sensors degrade? (exploratory)

Written and committed **before any degraded evaluation or training was run**.
This is an exploratory follow-up to the main result (no detectable map effect on
clean sensors, tuning and test). It uses the **tuning split only**. The test split
is not used for any part of this study, and `eval_scenes.py` refuses
`--degrade` together with `--split test`.

## Question

Does the HD-map stream lower prediction error once the other perception inputs
(ego and cooperative LiDAR and cameras) get worse?

## Conditions (fixed here, `sensor_degradation.py`)

These apply at load time to **every** agent's sensors, ego and every coop
neighbour alike. V2X messages (CAM `cam`, CPM `cpm_objects`) and map inputs
(`nearby_idx`, the lane graph) are left exactly as they are.

| X | Effect |
|---|---|
| `lidar_range` | drop LiDAR points with horizontal range `hypot(x, y) > 20 m` in that agent's sensor frame (if none are left, one all-zero point remains) |
| `cam_fog` | each view: `clamp(0.4 * img + 0.3 + N(0, 0.05²), 0, 1)` |
| `fog` | `lidar_range` + `cam_fog` |
| `sensors_off` | all LiDAR and camera tensors are zero, with their shapes kept |

The noise is deterministic. It is seeded by `zlib.crc32(f"{agent_id}_{scene}_{frame}")`
of the sensor's own agent, so it is identical across workers, runs and modes.
`verify_degrade.py` must pass on 50 samples before any evaluation.

## Step 1: clean-trained models under degradation

- Models: `D_seed{0,1,2}` (full) and `Dnomap_seed{0,1,2}` (no map), `checkpoint_epoch12.pt`,
  unchanged (the same checkpoints as the clean result).
- For each X: `eval_scenes.py --run-dir runs/<run> --ckpt runs/<run>/checkpoint_epoch12.pt --degrade X`
  writes `eval_<run>_tuning_X.log` (`run_degraded_evals.sh`), and then
  `analyze_seeds.py --suffix tuning_X`.
- Baseline for the "increase" numbers: the existing clean tuning logs `eval_<run>_e12.log`.

## Step 2: training under degradation

The training condition is fixed as **`fog`**. Full and no-map models are trained with the
same flags as D / Dnomap plus `--degrade fog`, seeds 0, 1 and 2, then evaluated at epoch 12 on
the tuning split under `fog`. The rule below applies unchanged. No other training
condition will be added after the step 1 results are seen.

## Rule (applied separately to each condition X, and to step 2)

**The map helps under condition X** only if both of these hold:

1. the seed-averaged full minADE6 is lower than the seed-averaged no-map minADE6 in **≥ 7 of the 8** tuning scenes, **and**
2. the ALL minADE6 gap (no-map − full, seed means) is **> 2 × max(seed std of full, seed std of no-map)**.

These two conditions are the clean study's rule, with the same thresholds, as implemented in
`analyze_seeds.py`. If either fails, the result for X is "no detectable map effect under X".

## Reporting

- All four step 1 conditions and step 2 are reported, whatever the outcome.
- The ALL minADE6 and ADE1 increase over clean, for full and for no-map, is
  **descriptive only**. It is not a decision criterion. A larger increase for no-map than
  for full is not, on its own, a claim that the map helps.
- ADE1 per-scene wins are printed but are not part of the rule.
- `verify_degrade.py` and every degraded eval log print how many sensors (ego + coop) hit the
  empty-LiDAR fallback, out of how many.
- `sensors_off` is a bound (map + V2X messages only). It is read with the same rule.
- These results are exploratory and tuning-split only. They do not change the main clean
  test-split conclusion.
