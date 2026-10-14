# Decision rules: RSU (V2I) extension for Part 2

Status: pre-registered before any RSU coverage or training result was seen.
Commit hash of this file is recorded in git history.

## Setup
- Task: Part 2 environment prediction (predict other vehicles), option B1.
- split_version=v2, map_transform=mirror, test scenes from test_scenes_v2.json.
- Models: E_V2X (existing, 3 seeds) vs E_V2X+RSU (new, seeds 0, 1, 2).
- E_V2X+RSU: 12 epochs, seeds 0, 1, 2, checkpoint = final model after epoch 12
  (no best-epoch selection), all other hyperparameters and flags identical to E_V2X
  in run_env_seeds.sh. The only change is the added RSU stream (agent 0: 4 cameras +
  LiDAR, image cache) and RSU-inclusive visibility.
- All verdicts on the tuning split first; test split only through the guard, once.

## Precondition: RSU point counts
- The mismatch between recomputed RSU in-box LiDAR point counts and the dataset's
  num_lidar_pts must be explained before coverage is measured.
- Pass bar: recomputed and dataset counts agree exactly for >= 95% of RSU boxes, or
  the cause is identified and documented (e.g. frame/handedness, sweep timing).
- If neither holds, RSU visibility is defined from the recomputed counts, and this is
  stated as a limitation. This decision is recorded here before H1 is computed.

## Target sets
- Visibility threshold unchanged: >= 5 LiDAR points.
- V: kept targets visible to the ego or another CAV (the E_V2X set).
- R: kept targets visible to the RSU and to no CAV (RSU-only).
- Moving / stationary split as in DECISION_RULES_ENV.md.

## H1: coverage (deterministic, no seeds)
- Coverage = share of kept targets in V, and in V union R.
- The RSU adds coverage if V union R exceeds V by >= 2 percentage points (tuning split).
- Also reported per scene and for moving targets (descriptive).

## H2: RSU-only vehicles vs constant velocity
- Set: moving targets in R. Metric: top-1 ADE1.
- Eligible scenes: >= 20 moving R targets. Minimum 6 eligible scenes, else H2 is
  reported as "not testable".
- Helps if E_V2X+RSU seed-mean ADE1 is lower than constant velocity in at least
  ceil(7/8 x eligible scenes) scenes AND the pooled gap exceeds 2x the model's seed std.

## H3: E_V2X+RSU vs E_V2X on moving targets
- Set: moving targets in V only (visible to both models), so accuracy is not mixed
  with coverage. Metric: minADE6.
- Helps if E_V2X+RSU seed-mean minADE6 is lower in >= 7/8 scenes AND the pooled gap
  exceeds 2x the larger seed std of the two models.
- Otherwise: no detectable effect (either direction).

## Descriptive only (no verdicts)
- ALL and stationary targets; ADE1 on V; minADE6 on R; per-scene tables.

## Test evaluation
- After all tuning verdicts are written into this file's results section, run one
  guarded test evaluation of the 6 checkpoints (3 x E_V2X+RSU, existing 3 x E_V2X
  results reused if already evaluated under the same code). Same rules apply; no
  rule changes after tuning results.

## Results
(empty until runs finish)
