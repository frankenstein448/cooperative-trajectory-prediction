# HD Map + V2X Fusion for Trajectory Prediction

Master's thesis project: "Integration of HD Map and V2X Data for Enhanced Environment Prediction in Autonomous Driving"
TU Chemnitz, Chair of Computer Engineering

## Overview

Predicts vehicle trajectories at intersections by fusing four sources of information per agent:

1. **Ego sensors** — the vehicle's own LiDAR (PointNet-style encoder) + 3-camera front view (small CNN)
2. **Cooperative (RSU/CAV) sensors** — same sensor types, from nearby agents within V2X communication range
3. **V2X messages** — CAM (position/heading/speed) + CPM (nearby detected objects), built from dataset metadata
4. **HD map** — lane graph encoded with an R-GCN, queried via cross-attention for each agent's locally relevant lanes

All four streams are fused and passed to a multimodal trajectory head (6 candidate future paths + confidence scores), trained with a winner-takes-all loss.

Built on the [V2X-Sim 2.0](https://ai4ce.github.io/V2X-Sim/) dataset and CARLA-exported HD map topology.

## Status

- All four streams implemented and verified working
- Ground truth trajectory extraction (ego-centric, from future agent positions)
- Winner-takes-all loss (min-ADE mode selection + regression + confidence classification)
- Pilot training pipeline with train/val split, running on Town05 scenes (38 scenes, ~10.6k samples)
- Stream encoders currently frozen during training (only the fusion + trajectory head layers are trained)

## Contents

**HD map pipeline**
- `export_topology.py` — exports lane topology from CARLA maps
- `topology/` — exported lane connection data (Town03, Town04, Town05)
- `rgcn_map_encoder.py` — R-GCN lane graph encoder
- `map_cross_attention_fusion.py` — per-agent cross-attention over nearby lane nodes
- `coord_transform.py` — V2X-Sim → CARLA coordinate alignment (Town05 only, fitted via grid search)
- `grid_search_transform.py` — coordinate alignment fitting process

**Sensor streams**
- `ego_sensor_encoder.py` — LiDAR PointNet-style encoder
- `ego_camera_encoder.py` — camera CNN encoder (3 front views)
- `ego_stream.py` — combined ego (LiDAR + camera) embedding
- `coop_sensor_stream.py` — cooperative RSU/CAV sensor stream (reuses ego encoder, mean-pooled across neighbors)
- `find_coop_neighbors.py` — real neighbor-finding by V2X communication range
- `v2x_message_stream.py` — CAM + CPM message stream, plus the shared V2X-Sim metadata loader

**Fusion, training, evaluation**
- `four_stream_fusion.py` — top-level fusion module + multimodal trajectory head
- `ground_truth_extractor.py` — extracts each agent's real future trajectory
- `trajectory_loss.py` — winner-takes-all loss
- `pilot_dataset.py` — PyTorch Dataset over Town05 scenes
- `train.py` — training loop with train/val split

## CARLA replay video (visualisation only)

Recorded V2X-Sim data of one tuning-split scene replayed in CARLA 0.9.15 (`~/carla_sim`), with the cached seed-0
epoch-12 predictions of E_V2X and E_EgoOnly drawn over the render (no training, no test split). End to end, from the
data checkout:

```
~/hd-maps-dev/make_carla_replay.sh [--scene 16] [--agent 4] [--still-frame 35]
```

- step 0: predictions cache `viz_env/pred/` via `render_env_bev.py` (only if missing), gpu-test-venv
- step 1: `export_carla_replay.py` -> `viz_env/carla_replay/scene<S>_ego<A>/`, gpu-test-venv
- step 2: `carla_replay.py render` -> `..._render/`, carla-venv; starts and stops CARLA itself
- step 3: `overlay_carla_replay.py --all` -> `..._overlay/` (frames, mp4 at 2.5 fps, 300 dpi still), gpu-test-venv

Each step refuses to start while a training / eval run, another CARLA server or a GPU compute process is running,
prints its exact command and stops the script on error.

## Dataset

Raw data (`V2X-Sim-2.0/`) and CARLA map assets (`*.xodr`, `carla_sim/`) are not included in this repo — see `.gitignore`. Topology `.pkl` files (small, derived from `.xodr`) are included.

## Next steps

- Multi-town support (Town03/Town04 coordinate alignment)
- Unfreeze stream encoders for end-to-end training
- Speed/scaling optimization for larger training runs
- Evaluation metrics (ADE/FDE, collision rate) for comparison against published baselines
