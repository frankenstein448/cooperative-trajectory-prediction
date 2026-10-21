# archive/

Files that no thesis result, table or figure depends on, kept for the record (repository audit, 2026-10-21).
Nothing outside `archive/` imports, calls or reads them. They are kept unchanged and are not maintained: scripts may
need the repository root on `PYTHONPATH`, a live CARLA server, or paths that no longer exist. Legacy code that
Appendix G needs (pilot pipeline, transform grid search / refit / residual / heading scripts, `test_transforms*.py`,
the legacy branch of `coord_transform.py`, split v1 in `scene_split.py`) is **not** here; it stays in the root.

33 files: 22 moved from the repository (`git mv`, history kept), 11 copied from the untracked files of the main
checkout.

## Moved from the repository root (15)

| file | what it was | why archived |
|---|---|---|
| `check_calibration.py` | prints the calibrated sensor of one BEV_TOP sample | broken: looks up `sweeps/BEV_TOP_id_1/scene_77_000005.npz`, which is not in the dataset |
| `check_multiple_frames.py` | prints ego poses of a few frames of scene 77, agent 1 | August exploration, no caller |
| `check_node_spacing.py` | nearest-neighbour spacing of the Town05 topology graph nodes | August exploration, no caller |
| `check_true_nearest_node.py` | nearest topology node to one hand-picked Town05 CARLA point | August exploration, no caller |
| `visualize_nearby_lanes.py` | plots topology lanes near one agent (Town05, agent 1) | August exploration, no caller; wrote `Town05_agent1_nearby_lanes.png` |
| `find_transform.py` | first V2X-Sim -> CARLA transform probe against a live Town05 | earliest transform probe; superseded by `grid_search_transform.py` (Appendix G.3, kept) |
| `check_transformed_waypoint.py` | nearest CARLA waypoint to one transformed point (live Town05) | earliest transform probe, no caller |
| `check_waypoint.py` | loads Town05 in CARLA and queries a waypoint | earliest transform probe, no caller |
| `find_good_test_point.py` | picks a scene-77 agent point for the transform probes (live Town05) | earliest transform probe, no caller |
| `get_lidar_position.py` | world position of one LIDAR_TOP sample (scene 77) | earliest transform probe, no caller |
| `profile_getitem.py` | timing breakdown of `PilotDataset.__getitem__` | one-off performance tool, no caller |
| `profile_train_step.py` | timing of ~50 train batches (loader wait, transfer, step) | one-off performance tool, no caller |
| `diag_lidar_scene30_frame19_ego3.png` | first output of `diag_lidar_behind.py` (two panels) | image; the thesis figure is `viz_env/thesis_figures/lidar_blindspot.pdf` (`--print`) |
| `Town05_agent1_nearby_lanes.png` | output of `visualize_nearby_lanes.py` | image of an archived script |
| `Town05_dense_waypoints.png` | plot of the Town05 dense waypoint cloud | image; the data is `dense_waypoints_town05.npy` |

## Moved from `topology/` (7)

| file | what it was | why archived |
|---|---|---|
| `topology/inspect_ego_pose.py` | prints the first `ego_pose.json` / `scene.json` records | metadata probe, no caller |
| `topology/inspect_sample_data.py` | prints the first `sample_data.json` record and its channels | metadata probe, no caller |
| `topology/inspect_sample_data_full.py` | same, all fields of the first record | metadata probe (near-duplicate of the above), no caller |
| `topology/check_coordinate_range.py` | x / y range of the Town05 topology graph | metadata probe, no caller |
| `topology/get_agent_world_pos.py` | world position of agent 1 in scene 77, frame 23 | metadata probe, no caller (`visualize_predictions.get_agent_world_pos` is a different function) |
| `topology/extract_agent_pos.py` | agent positions from a coperception `V2X-Sim-det` sample | reads `~/V2X-Sim-det`, which no thesis script uses |
| `topology/RGCN map encoder` | early copy of `rgcn_map_encoder.py` (20 lines differ; no `.py`, spaces in the name) | superseded by `rgcn_map_encoder.py` |

## Copied from untracked files of the main checkout (11)

| file | what it was | why archived |
|---|---|---|
| `coord_transform.py.bak_final` | `coord_transform.py` with the per-town rotations of 2026-10-05 (Town04 theta 260.50) | backup copy; the values are in `coord_transform.py`'s legacy branch |
| `coord_transform.py.bak_town05_fixed` | earlier version, Town04 theta 296.50 active | backup copy; the only record of that candidate as active code |
| `train.py.bak` | `train.py` of 2026-10-06 (124 lines) | backup copy of an old training loop |
| `train.py.bak_before_rename` | `train.py` of 2026-10-08, before a rename | backup copy |
| `train.py.bak_random_split` | `train.py` with the pilot's random split | backup copy (pilot, Appendix G.1 uses the kept legacy scripts) |
| `train_ablation.py.bak_random_split` | `train_ablation.py` with the random split | backup copy |
| `eval_by_town.py.bak` | `eval_by_town.py` before 2026-10-08 (13 lines differ) | backup copy |
| `eval_scenes.py.bak_20261009_1527` | first `eval_scenes.py` (46 lines) | backup copy |
| `auto_loss_after_nocoop.sh` | one-off chain: wait for the NoCoop verdict, merge c09a06d, verify, dry run, `run_loss_seeds.sh` | one-off orchestration of the loss study; the study itself is `run_loss_seeds.sh` |
| `train_log_20260813_1910.txt` | pilot training log (random split, 30 epochs) | pilot log of 2026-08 |
| `training_history.txt` | terminal capture of an early pilot training session | pilot log of 2026-08 |
