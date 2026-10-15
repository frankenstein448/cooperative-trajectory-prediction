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

## Precondition outcome (recorded 14 Oct 2026, before Step 2)
- Matching variant: LIDAR_TOP, raw RSU pose and extrinsic (no mirror), keyframe sweep.
  Recount equals num_lidar_pts[0] for 6,451 / 6,451 RSU boxes with width > 0
  (scenes 4, 6, 3, all 100 frames). Index 0 of num_lidar_pts is the RSU.
- Cause of the mismatch: zero-width boxes (66.8% of bicycles; 8.5% of all annotations,
  8.1% of kept env targets). Their dataset counts reach ~290 m while stored sweeps
  reach ~70 m, so these counts cannot come from the stored point cloud.
- Pass condition met: cause identified and documented.
- Decision: visibility stays num_lidar_pts >= 5 for all agents including the RSU,
  identical to E_V2X, so set V is unchanged and H3 stays comparable. The H1 verdict
  uses all kept targets; H1 excluding zero-width targets is reported as a
  sensitivity check (descriptive, no verdict). Zero-width targets are also reported
  separately in H2/H3 tables.

## Results
### H1 coverage (14 Oct 2026, computed after rules commit e18347b)
- Tuning, all kept targets: V 84.33% -> V+R 89.29%, gain +4.97 pp >= 2 pp.
  VERDICT H1: the RSU adds coverage.
- Sensitivity, zero-width targets excluded: +5.13 pp (descriptive).
- Tuning moving: +4.88 pp. Test v2 (descriptive): all +4.20 pp, moving +3.89 pp.
- Largest gain Town04 (tuning moving +11.1 pp, test moving +13.3 pp); smallest test Town03 (+1.5 pp).

### Implementation decisions (14 Oct 2026, before any E_V2X+RSU training)
- H2 eligibility (counts only): tuning 7/8 scenes eligible (scene 21 < 20 moving R targets),
  so H2 requires 7/7 wins; test 8/8 eligible. Eligible scenes hold only 5-30 distinct
  RSU-only vehicles each (correlated samples; noted as a limitation).
- H2/H3 pooled gap: pooled over all moving targets of the set in the split, including
  ineligible scenes, as in env_metrics.rule for Part 2.
- env_norm for E_V2XRSU: computed over its own known train targets (V union R), the same
  convention as all existing env configs.
- Run names: E_V2XRSU_seed{0,1,2}.

### H2/H3 tuning verdicts (15 Oct 2026; analyze_rsu.log, E_V2XRSU seeds 0-2, epoch 12)
- H2 RSU-only moving vs constant velocity (ADE1): 6.544 +/- 0.366 vs 9.885, wins 6/7 eligible
  scenes (7 needed; scene 30 lost: 6.535 vs 5.388), pooled gap +3.341 > 0.731.
  VERDICT H2: no detectable effect.
- H3 E_V2XRSU vs E_V2X, moving V (minADE6): 3.019 +/- 0.066 vs 2.928 +/- 0.062, wins 2/8,
  gap -0.091. VERDICT H3: no detectable effect.
- Descriptive: top-1 on moving V worse with RSU (ADE1 7.480 +/- 0.088 vs 6.411 +/- 0.120;
  miss1 0.696 vs 0.645). Not a pre-registered hypothesis.
