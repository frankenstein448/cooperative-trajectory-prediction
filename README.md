# HD Map + V2X Fusion for Trajectory Prediction

Code of the Master's thesis "Integration of HD Map and V2X Data for Enhanced Environment Prediction in Autonomous
Driving", TU Chemnitz, Chair of Computer Engineering.

The thesis studies whether an HD map and V2X data (CAM / CPM messages and the sensors of cooperating vehicles and a
road-side unit) improve 6 s, 6-mode trajectory prediction on V2X-Sim 2.0 (CARLA Town03 / 04 / 05, 5 Hz). Part 1 (ego
task) predicts the ego vehicle's own future; Part 2 (environment task) predicts the surrounding vehicles, including
those only a cooperating CAV or the RSU sees. Every comparison was fixed in a pre-registered decision rule
(`runs/DECISION_RULES_*.md`) before its runs, decided on the tuning split, and confirmed once on a guarded test split.

> **WARNING - the test split is guarded and must not be re-evaluated.** Every test evaluation goes through
> `final_test_guard.py`, which records it in `runs/TEST_EVALS.log` and refuses a second evaluation of the same
> checkpoint. The final test (`run_final_test.sh`, `run_rsu_final_test.sh`, `run_mapv2_ego_final_test.sh`) is done;
> its logs are in this repository. Do not run these scripts again, and do not call `eval_scenes.py` / `eval_env.py`
> with `--split test`.

## Repository layout

| path | content |
|---|---|
| `*.py` (root) | all code; one row per file in **`FILES.md`** (group, thesis section, caller) |
| `run_*.sh` | pre-registered experiment chains (training, tuning evaluation, rule, guarded test) |
| `make_bev_figures.sh`, `make_carla_replay.sh`, `make_thesis_tables.py` | figures, videos and LaTeX tables |
| `runs/DECISION_RULES_*.md`, `runs/DESIGN_ENV.md` | pre-registered rules and the environment-task design |
| `runs/TEST_EVALS.log` | the guard's record of every test-split evaluation |
| `runs/<run>/config.json`, `cam_norm.json`, `env_norm.json`, `history_norm.json` | run settings and input normalisation (checkpoints are not in git) |
| `analyze_*.log`, `eval_*.log`, `train_*.log`, `verify_*.log`, `diag_*.log`, `check_*.log`, `build_*.log`, `export_*.log` | evidence: the logs of every run, rule and check |
| `lane_polylines_town0*.npz`, `dense_waypoints_town0*.npy`, `junction_waypoints_town0*.npy`, `topology/*_topology.pkl` | exported CARLA map data (map v2, off-road metric, junctions, map v1 graph) |
| `test_scenes.json`, `test_scenes_v2.json`, `scene_split_proposal.json` | test scenes of split v1 / v2 and the v2 proposal |
| `requirements_gpu.txt`, `requirements_carla.txt` | `pip freeze` of the two virtual environments |
| `archive/` | files no thesis result depends on; one line each in **`archive/README.md`** |

## Setup

**Virtual environments** (Ubuntu, NVIDIA GPU for training):

| venv | Python | requirements | used for |
|---|---|---|---|
| `gpu-test-venv` | 3.12.3 | `requirements_gpu.txt` (torch 2.13.0, torch-geometric 2.8.0.post1, numpy 2.5.1) | everything except the CARLA client |
| `carla-venv` | 3.10.20 | `requirements_carla.txt` (carla 0.9.15) | map export and the CARLA replay render |

```
python3.12 -m venv ~/gpu-test-venv && ~/gpu-test-venv/bin/pip install -r requirements_gpu.txt
python3.10 -m venv ~/carla-venv    && ~/carla-venv/bin/pip install -r requirements_carla.txt
```

A third environment on the development machine, `coperception-venv` (Python 3.7, the coperception toolkit), was
only used to inspect V2X-Sim at the start; no thesis script uses it.

**V2X-Sim 2.0**: download from the dataset's site (https://ai4ce.github.io/V2X-Sim/). Expected layout:

```
$V2X_SIM_ROOT/                 default ~/V2X-Sim-2.0
  v2.0/                        sample_data.json, ego_pose.json, sample_annotation.json, calibrated_sensor.json, ...
  sweeps/                      LIDAR_TOP_id_<a>/, SEMLIDAR_TOP_id_<a>/, CAM_FRONT*_id_<a>/, CAM_id_0_<k>/ (RSU), ...
  maps/
```

**CARLA 0.9.15** (map export and the replay video only), installed in `$CARLA_ROOT` (default `~/carla_sim`).
`client.load_world()` crashes the server on the development machine, so the town is set as the startup map in
`CarlaUE4/Config/DefaultEngine.ini` (restore it afterwards) and the server is started with
`./CarlaUE4.sh -RenderOffScreen -opengl`. `carla_replay.py` does this itself (it edits `DefaultEngine.ini` for the
start and restores it from its backup `DefaultEngine.ini.orig_replay`).

**Paths** (`paths.py`): every location can be overridden by an environment variable; the defaults are those of the
development machine.

| variable | default | content |
|---|---|---|
| `V2X_SIM_ROOT` | `~/V2X-Sim-2.0` | dataset (`v2.0/`, `sweeps/`) |
| `HD_CACHE` | `~/hd_cache` | caches: `images/`, `images_rsu/`, `env_targets/` |
| `CARLA_ROOT` | `~/carla_sim` | CARLA install (OpenDRIVE maps for `export_junctions.py`) |

All scripts run from the repository root, which is also the data working directory (`runs/`, logs, `viz_env/` are
written there). `make_bev_figures.sh` and `make_carla_replay.sh` expect that directory at `~/hd-maps`.

## Pipeline

1. **Map export** (5.2; carla-venv, CARLA started with the town as startup map; outputs are committed):
   `python dense_waypoints.py Town03`, `python export_lane_polylines.py Town03 --include-parking` (and Town04 / 05),
   `python export_junctions.py` (offline, from the OpenDRIVE files), `python export_topology.py`.
2. **Caches** (5.3; gpu-test-venv): `python build_image_cache.py`, `python build_rsu_image_cache.py`,
   `python build_env_target_cache.py`; checked by `verify_image_cache.py`, `verify_rsu_image_cache.py`,
   `verify_env_targets.py`.
3. **Training and tuning evaluation** (gpu-test-venv; each chain trains seeds 0-2 for 12 epochs, evaluates epoch 12 on
   the tuning split and applies its rule): `./run_corrected_seeds.sh` (ego task C_D, C_Dnomap, C_EgoOnly),
   `./run_mapv2_ego_seeds.sh` (C_Dmapv2), `./run_env_seeds.sh` (E_EgoOnly, E_V2Xnomap, E_V2X), `./run_rsu_seeds.sh`
   (E_V2XRSU).
4. **Rules on the tuning split**: `analyze_seeds.py`, `analyze_conditions.py`, `analyze_env.py`, `analyze_rsu.py`
   (called by the chains; their logs are `analyze_*.log`).
5. **Guarded final test** (done, do not re-run): `run_final_test.sh`, `run_mapv2_ego_final_test.sh`,
   `run_rsu_final_test.sh` -> `*_test_final.log`, `analyze_*final_test.log`, `runs/TEST_EVALS.log`.

## Thesis section -> script -> output

Logs are in the repository root unless a path is given. Studies on split v1 / legacy transform (App. E, App. G.2) use
runs whose `config.json` has no `split_version` / `map_transform` key.

| section | script / command | output |
|---|---|---|
| 4.4 conversion | `coord_transform.py` (mirror; legacy branch kept), `python verify_map_transform.py` | `verify_map_transform.log` |
| 4.5 split, protocol, guard | `scene_split.py` (v2), `python check_scene_overlap.py [--split-version v2]`, `python check_scene_components.py`, `python verify_split_v2.py`; `final_test_guard.py`, `python test_final_test_guard.py`; `runs/DECISION_RULES_*.md` | `check_scene_overlap.log`, `check_scene_overlap_v2.log`, `check_scene_components.log`, `scene_split_proposal.json`, `verify_split_v2.log`, `runs/TEST_EVALS.log` |
| 4.6.1 alignment | `python check_map_alignment.py [--mirror-check]`, `python check_offroad_gt.py` | `check_map_alignment.log`, `check_map_alignment_mirror.log`, `check_offroad_gt.log` |
| 4.6.2 model inputs | `verify_cam_norm.py`, `verify_drop_cam_xy.py` | `verify_drop_cam_xy.log` |
| 4.6.3 annotations / sensors | `python check_zero_speed.py`; `python diag_lidar_behind.py --print --out viz_env/thesis_figures/lidar_blindspot.pdf` | `check_zero_speed.log`; Fig. 4.2 `viz_env/thesis_figures/lidar_blindspot.pdf` |
| 4.7 ego task (4.7.3 map v1, 4.7.4 map v2) | `train.py --task ego` (flags: `run_corrected_seeds.sh`, `run_mapv2_ego_seeds.sh`); `python test_get_nearby_lane_nodes.py`; `verify_mapv2_tokens.py`, `verify_mapv2_off.py` | `verify_mapv2_tokens.log`, `verify_mapv2_off.log` |
| 4.8 environment task | `train.py --task env` (flags: `run_env_seeds.sh`); `verify_env_model.py`, `python test_env_stage2.py` | `verify_env_model.log` |
| 4.9 RSU | `train.py --task env --use-rsu` (flags: `run_rsu_seeds.sh`); `verify_rsu_off.py` | `verify_rsu_off.log` |
| 5.2 map export | pipeline step 1 | `dense_waypoints_town0*.npy`, `lane_polylines_town0*.npz`, `junction_waypoints_town0*.npy`, `topology/*.pkl`, `export_*.log` |
| 5.3 data preparation, caches | pipeline step 2 | `$HD_CACHE/images/`, `images_rsu/`, `env_targets/`; `build_*.log`, `verify_env_targets.log`, `verify_image_cache.log`, `verify_rsu_image_cache.log` |
| 5.5 evaluation, guard, visualisation | `eval_scenes.py`, `eval_env.py` (metrics: `env_metrics.py`, `ranking_metrics.py`, `safety_metrics.py`, `scene_conditions.py`), `final_test_guard.py`; `./make_carla_replay.sh` | `eval_*.log`, `runs/<run>/eval_env_*_checkpoint_epoch12.npz`; `viz_env/carla_replay/` |
| 6.1 data validation | `python check_offroad_gt.py` (Table 6.2); `diag_lidar_behind.py` | `check_offroad_gt.log`; `lidar_blindspot.pdf` |
| 6.2 ego task | `./run_corrected_seeds.sh`, `./run_mapv2_ego_seeds.sh`; test: `run_final_test.sh`, `run_mapv2_ego_final_test.sh` (done); 6.2.1 V2X: `verify_v2x_ablation.py`; `python make_thesis_tables.py` | `eval_C_*_e12.log`, `analyze_seeds_corrected_map.log`, `analyze_seeds_corrected_v2x.log`, `analyze_conditions_corrected.log`, `analyze_seeds_mapv2_ego.log`, `analyze_conditions_mapv2_ego.log`; `eval_C_*_test_final.log`, `analyze_seeds_final_test_map.log`, `analyze_seeds_final_test_v2x.log`, `analyze_conditions_final_test.log`, `analyze_*mapv2_ego_final_test.log`; `viz_env/thesis_tables/ego_per_scene_{tuning,test}.{tex,csv}` |
| 6.3 environment task | `./run_env_seeds.sh`; test: `run_final_test.sh` (done); `python make_thesis_tables.py` | `eval_env_E_*_e12.log`, `analyze_env.log`, `eval_env_E_*_test_final.log`, `analyze_env_final_test.log`; `viz_env/thesis_tables/env_per_scene_*`, `env_subsets_*`, `env_coverage.{tex,csv,pdf}`, `town04_30_failure.{tex,csv}` |
| 6.4 RSU | `./run_rsu_seeds.sh`; test: `run_rsu_final_test.sh` (done); 6.4.4: `diag_rsu_sensitivity.py` (Table 6.12) | `eval_env_E_V2XRSU_*`, `analyze_rsu.log`, `analyze_rsu_final_test.log`, `diag_rsu_E_V2XRSU_seed*.log` |
| 6.6 qualitative | `./make_bev_figures.sh` (`render_env_bev.py`) | `viz_env/scene*_ego*_vs_*/` (mp4, `stills/`), `viz_env/thesis_figures/bev_*.pdf` |
| App. D figures | `./make_bev_figures.sh` (all-modes PDFs, #22); `./make_carla_replay.sh` (Fig. D.5) | `viz_env/thesis_figures/*_allmodes.pdf`; `viz_env/carla_replay/scene16_ego4_overlay/` |
| App. E development studies (split v1, legacy transform) | E.1.2: `verify_cam_norm.py`, `verify_drop_cam_xy.py`, `verify_relaxed_wta.py` (configuration E); E.1.3: `diag_speed_sensitivity.py` (Table E.3); E.2.1: `run_history_seeds.sh`, `verify_history.py`, `diag_history_sensitivity.py`; E.2.2: `run_loss_seeds.sh`, `run_wta_soft_seeds.sh`, `verify_loss.py`; E.2.3: `run_v2x_ablation.sh`, `verify_v2x_ablation.py`; E.2.4: `run_degraded_evals.sh`, `run_fog_training.sh`, `sensor_degradation.py`, `verify_degrade.py`; E.2.5: `diag_map_sensitivity.py` | `verify_drop_cam_xy.log`, `verify_relaxed_wta.log`, `diag_speed_sensitivity.log`, `analyze_seeds_history.log`, `verify_history.log`, `analyze_seeds_loss.log`, `analyze_seeds_wta_soft.log`, `verify_loss*.log`, `analyze_seeds_NoCPM.log`, `analyze_seeds_NoCoop.log`, `analyze_seeds_EgoOnly.log`, `analyze_seeds_tuning_*.log`, `analyze_seeds_Ffog_tuning_fog.log`, `verify_degrade.log`, `diag_map_D_seed*.log` |
| App. G.1 random split (pilot) | `evaluate.py`, `eval_metrics.py`, `eval_ablation.py`, `eval_by_town.py`, `train_ablation.py`, `diag_scores.py` (pilot checkpoints `checkpoints_e2e*/`, not in git) | `diag_scores_E.log`, `diag_scores_full.log` |
| App. G.2 split v1 (legacy) | `run_seeds.sh`, `run_test_evals.sh` (`analyze_seeds.py --suffix test`), `analyze_seeds_test.py`, `run_condition_evals.sh`, `run_turning_test.sh`, `run_safety_reeval.sh` | `eval_{D,Dnomap,EgoOnly}_seed*_e12.log`, `eval_{D,Dnomap}_seed*_test.log`, `eval_*_e12_cond.log`, `eval_*_test_cond.log`, `eval_*_safety.log`, `analyze_seeds.log`, `analyze_seeds_test.log`, `analyze_conditions.log`, `analyze_turning_test.log` |
| App. G.3 legacy transforms | `grid_search_transform.py`, `refine_town03_theta.py`, `refine_town04_theta.py`, `refit_transform.py`, `refit_town04_constrained.py`, `score_transforms.py`, `check_town0*_residuals*.py`, `check_residual_distribution.py`, `check_heading_*.py`, `plot_alignment.py`, `test_transforms*.py` (live CARLA) | `refit.log`, `run_town04.log`; `coord_transform.TOWN_TRANSFORMS` |

**Figures and videos** (`make_bev_figures.sh`): four BEV videos (mp4 at 2.5 fps, 300 dpi stills) - Town05 #22 ego 1
and #16 ego 4, Town04 #30 ego 3, each E_V2X vs E_EgoOnly, and Town04 #30 ego 3 E_V2X+RSU vs E_V2X - plus the eight
`--print` PDFs and `lidar_blindspot.pdf`. Top-1 ends in an arrowhead (a dot with a white rim when the top-1 path is
under 1 m), the other 5 modes of moving targets are thin lines, every line has a thin white outline.
**CARLA replay** (`make_carla_replay.sh [--scene 16] [--agent 4] [--still-frame 35]`): stage 0
`export_carla_replay.py` -> `viz_env/carla_replay/scene<S>_ego<A>/`, stage 1 `carla_replay.py render` (carla-venv;
starts and stops CARLA) -> `..._render/`, stage 2 `overlay_carla_replay.py --all` -> `..._overlay/` (frames, mp4,
300 dpi still). Both scripts refuse to start while a training / eval run or a GPU compute process is running, print
each command and stop on the first error. Visualisation only: tuning split, cached seed-0 epoch-12 predictions.

## Evidence

- **Pre-registered rules**: `runs/DECISION_RULES_*.md` (11 files). The first commit of each rule file is older than
  the result logs it governs (file times of the logs); later commits only record verdicts or the test evaluation,
  as the commit messages say.
- **Test evaluations**: `runs/TEST_EVALS.log` (written by `final_test_guard.py`), `*_test_final.log`.
- **Logs**: `analyze_*.log` (rules and verdicts), `eval_*.log` (per-run evaluations), `train_*.log`, `verify_*.log`
  (pre-training checks), `diag_*.log`, `check_*.log`.

## Large outputs (not in git)

| item | size | where |
|---|---|---|
| model checkpoints `runs/<run>/*.pt` (66 runs, epochs 1-12) | 7.4 GB | not on the submission medium; kept on the development machine |
| per-target evaluation results `runs/<run>/eval_env_{tuning,test}_checkpoint_epoch12.npz` (24 files) | 169 MB | submission medium: `results/results_npz.zip` with `results/SHA256SUMS.txt` (unzip in the repository root) |
| `viz_env/thesis_figures/`, `viz_env/thesis_tables/` | 1.2 MB | submission medium: `viz_env/thesis_figures/`, `viz_env/thesis_tables/` |
| BEV videos and stills, CARLA replay video and stills | 27 MB | submission medium: `viz_env/scene*_ego*_vs_*/`, `viz_env/carla_replay/scene16_ego4_overlay/` |
| caches (`$HD_CACHE`), V2X-Sim 2.0, CARLA, venvs | > 200 GB | not on the medium; rebuilt with the setup and pipeline above |

The submission medium also holds the repository as a git bundle (all branches and tags) and as a plain export of
`main` at the tag `thesis-submission`; see its `SUBMISSION_README.txt`.
