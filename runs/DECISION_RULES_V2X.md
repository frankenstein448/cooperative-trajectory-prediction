# Decision rules: what do the V2X inputs contribute?

Written and committed **before any V2X ablation training was run**. The study uses the
**tuning split only**. The test split is not used for any part of it.

## Question

Does each V2X component lower prediction error compared with the same model without it?

## Inputs (from the code, `pilot_dataset.py`, `v2x_message_stream.py`, `coop_sensor_stream.py`)

- **CPM objects**: every `sample_annotation` box within 50 m of the target agent at the
  current frame (ground-truth boxes, not detections), as `[M, 7]`: world-frame offset
  (x, y, not rotated), absolute z, size (3), absolute yaw. Encoded by `CPMEncoder`
  (per-object MLP, max-pool), concatenated with the CAM embedding in `V2XMessageStream`.
- **Cooperative sensors**: the LiDAR and 3 camera views of every other camera-equipped
  CAV (agents 1–4 minus the target) within 200 m, encoded by `CoopSensorStream` (an
  EgoStream-architecture encoder with its own weights) and mean-pooled into one stream.
  The RSU (agent 0) is not used.
- **Ego CAM**: the target agent's own state from its `ego_pose`. Under D's flags, the model sees
  normalized heading and motion_utils speed (x, y zeroed). **Kept in every configuration.**

## Configurations (removal, like `--no-map`)

| Config | Flags | Model change | Dataset change |
|---|---|---|---|
| D (full, existing) | none | none | none |
| `NoCPM` | `--no-cpm` | no `cpm_enc`; V2X fuse input 256 → 128 | `get_cpm` skipped |
| `NoCoop` | `--no-coop` | no `coop_encoder`; top fuse input 4 → 3 streams | no neighbour search or neighbour sensor loading |
| `EgoOnly` | `--no-cpm --no-coop` | both | both: ego sensors + ego CAM + map |

With both flags off the model is D's: `train.py` asserts the parameter count, and
`verify_v2x_ablation.py` strict-loads `runs/D_seed0/checkpoint_epoch12.pt` and checks each
ablation's parameter count, the dataset fields and a forward pass. It must pass before training.
The sample set is unchanged (2,208 tuning samples).

## Runs

D's flags (`--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy`, map on),
seeds 0, 1 and 2, in the order EgoOnly ×3, NoCPM ×3, NoCoop ×3 (`run_v2x_ablation.sh`).
Each is evaluated at `checkpoint_epoch12.pt` on the tuning split. The comparison runs are the
existing `D_seed0`, `D_seed1`, `D_seed2` (epoch 12, unchanged checkpoints).

## Rule (applied to each ablation config separately)

**The removed component helps** only if both of these hold:

1. D's seed-mean minADE6 is lower than the ablation's in **≥ 7 of the 8** tuning scenes, **and**
2. the ALL minADE6 gap (ablation − D, seed means) is **> 2 × max(seed std of D, seed std of the ablation)**.

These are the map rule's thresholds, implemented as `analyze_seeds.py --rule component`
(in its output, "full" = D and "no-map" = the ablation). If either fails, the result is
"no detectable effect of the removed component".

- `NoCPM` tests the CPM objects, `NoCoop` the cooperative sensors, and `EgoOnly` both together.

## Safety metrics (descriptive only)

Miss rate (top-1 and best-of-6 FDE > 2 m) and off-road rate (top-1 and all 6 modes; points
> 2.5 m from the nearest lane-centre waypoint) are reported for every run, from the
`SAFE` block of `eval_scenes.py`. D's numbers come from `run_safety_reeval.sh`
(`eval_D_seed*_e12_safety.log`, whose accuracy rows must match the original logs exactly).
Per-town off-road numbers are flagged UNRELIABLE where the ground-truth off-road rate in that
town exceeds 5%. None of these metrics is part of the rule.

## Repetition on the final model

The ablation will be repeated on the final model (with whatever of history, map v2, etc. is
adopted) using `run_v2x_ablation.sh` with that model's flags, a new run prefix and its own
seed runs as the comparison. **The same rule, with the same thresholds, applies unchanged.**

## Reporting

All three configs are reported whatever the outcome, with per-run and per-scene numbers.
