# FILES.md

Every `.py` file that stays in the repository (root; `topology/` now holds only the `*_topology.pkl` graphs), with its
group, the thesis section it supports and how it is reached. Archived files are listed in `archive/README.md`.

Groups: **core** (pipeline code), **check-test** (verification scripts and unit tests), **diagnostic** (read-only
analyses), **figures-tables**, **legacy-AppG** (superseded, kept so Appendix G can be reproduced: G.1 random split,
G.2 split v1 legacy, G.3 legacy transforms), **legacy** (superseded, no thesis result).
"manual" = an entry point run by hand (its log, if any, is in the repository root).

Thesis sections: 4.4 conversion, 4.5 split / protocol / guard, 4.6 data validation (4.6.1 alignment, 4.6.2 model
inputs, 4.6.3 annotations / sensors), 4.7 ego task (4.7.3 map v1, 4.7.4 map v2), 4.8 environment task, 4.9 RSU,
5.2 map export, 5.3 data preparation / caches, 5.5 evaluation / guard / visualisation, 6.1-6.6 results, App. D figures,
App. E development studies, App. G superseded results.

| file | group | thesis section | entry point or imported by |
|---|---|---|---|
| `coord_transform.py` | core | 4.4 (legacy branch: App. G.3) | `pilot_dataset.py`, `env_map.py`, `train.py`, `eval_scenes.py` |
| `scene_split.py` | core | 4.5 (v1: App. G.2) | `train.py`, `eval_scenes.py`, `eval_env.py`, `build_env_target_cache.py` |
| `final_test_guard.py` | core | 4.5, 5.5 | `eval_scenes.py`, `eval_env.py`; `run_final_test.sh`, `run_rsu_final_test.sh`, `run_mapv2_ego_final_test.sh` |
| `paths.py` | core | none (data / cache / CARLA locations) | `pilot_dataset.py`, `v2x_message_stream.py`, `env_dataset.py`, `build_*_cache.py`, check / verify scripts |
| `pilot_dataset.py` | core | 4.7 | `train.py`, `eval_scenes.py`, `env_dataset.py`, `build_*_cache.py` |
| `v2x_message_stream.py` | core | 4.7 | `pilot_dataset.py`, `end_to_end_model.py`, `env_model.py`, `build_env_target_cache.py` |
| `ground_truth_extractor.py` | core | 4.7 | `pilot_dataset.py`, `build_env_target_cache.py` |
| `find_coop_neighbors.py` | core | 4.7 | `pilot_dataset.py`, `build_image_cache.py` |
| `motion_utils.py` | core | 4.7 | `pilot_dataset.py`, `eval_scenes.py` |
| `ego_stream.py` | core | 4.7 | `end_to_end_model.py`, `env_model.py` |
| `ego_sensor_encoder.py` | core | 4.7 | `pilot_dataset.py`, `ego_stream.py` |
| `ego_camera_encoder.py` | core | 4.7 | `pilot_dataset.py`, `ego_stream.py` |
| `coop_sensor_stream.py` | core | 4.7 | `end_to_end_model.py`, `env_model.py` |
| `history_encoder.py` | core | 4.7 | `end_to_end_model.py`, `env_model.py`, `train.py` |
| `map_cross_attention_fusion.py` | core | 4.7 | `end_to_end_model.py` |
| `end_to_end_model.py` | core | 4.7 | `train.py`, `env_model.py` |
| `trajectory_loss.py` | core | 4.7 | `train.py`, `eval_scenes.py`, `end_to_end_model.py` |
| `rgcn_map_encoder.py` | core | 4.7.3 | `end_to_end_model.py`, `pilot_dataset.py` |
| `env_map.py` | core | 4.8, 4.7.4 | `env_dataset.py`, `env_model.py`, `pilot_dataset.py` (map v2 tokens) |
| `env_dataset.py` | core | 4.8 (`use_rsu`: 4.9) | `train.py`, `eval_env.py` |
| `env_model.py` | core | 4.8 (RSU stream: 4.9) | `train.py`, `eval_env.py` |
| `train.py` | core | 4.7, 4.8, 4.9 | `run_seeds.sh`, `run_corrected_seeds.sh`, `run_env_seeds.sh`, `run_rsu_seeds.sh`, `run_mapv2_ego_seeds.sh`, other `run_*_seeds.sh` |
| `sensor_degradation.py` | core | App. E.2.4 | `pilot_dataset.py`, `train.py`, `eval_scenes.py` |
| `dense_waypoints.py` | core | 5.2 | manual (live CARLA); its `.npy` output is checked by `export_lane_polylines.py`, `export_junctions.py` |
| `export_lane_polylines.py` | core | 5.2 | manual (writes `lane_polylines_town0*.npz`) |
| `export_junctions.py` | core | 5.2 | manual (writes `junction_waypoints_town0*.npy`, required by `run_condition_evals.sh`, `run_turning_test.sh`) |
| `export_topology.py` | core | 5.2 | manual (writes `topology/*_topology.pkl`) |
| `build_image_cache.py` | core | 5.3 | manual; imported by `build_rsu_image_cache.py` |
| `build_rsu_image_cache.py` | core | 5.3 | `run_rsu_seeds.sh` |
| `build_env_target_cache.py` | core | 5.3 | manual; imported by `verify_env_targets.py` |
| `eval_scenes.py` | core | 5.5 | every ego `run_*.sh` |
| `eval_env.py` | core | 5.5 | `run_env_seeds.sh`, `run_rsu_seeds.sh`, `run_final_test.sh`, `run_rsu_final_test.sh` |
| `env_metrics.py` | core | 5.5 | `eval_env.py`, `analyze_env.py`, `analyze_rsu.py` |
| `ranking_metrics.py` | core | 5.5 | `eval_scenes.py` |
| `safety_metrics.py` | core | 5.5 | `eval_scenes.py` |
| `scene_conditions.py` | core | 5.5 | `eval_scenes.py`, `analyze_conditions.py`, `env_metrics.py` |
| `visualize_predictions.py` | core | 5.5 | `eval_scenes.py` (`load_model`) |
| `analyze_seeds.py` | core | 6.2 | `run_corrected_seeds.sh`, `run_final_test.sh`, `run_mapv2_ego_seeds.sh`, other ego `run_*.sh` |
| `analyze_conditions.py` | core | 6.2 | `run_condition_evals.sh`, `run_corrected_seeds.sh`, `run_final_test.sh`, `run_turning_test.sh` |
| `analyze_env.py` | core | 6.3 | `run_env_seeds.sh`, `run_final_test.sh`, `make_thesis_tables.py` |
| `analyze_rsu.py` | core | 6.4 | `run_rsu_seeds.sh`, `run_rsu_final_test.sh` |
| `check_map_alignment.py` | check-test | 4.6.1, 4.4 | manual (`check_map_alignment.log`, `check_map_alignment_mirror.log`) |
| `verify_map_transform.py` | check-test | 4.4, 4.6.1 | manual (`verify_map_transform.log`) |
| `verify_split_v2.py` | check-test | 4.5 | manual (`verify_split_v2.log`) |
| `check_scene_overlap.py` | check-test | 4.5 | manual (`check_scene_overlap.log`, `check_scene_overlap_v2.log`) |
| `check_scene_components.py` | check-test | 4.5 | manual (`check_scene_components.log`) |
| `test_final_test_guard.py` | check-test | 4.5, 5.5 | manual (unit test) |
| `verify_cam_norm.py` | check-test | 4.6.2, App. E.1.2 | manual |
| `verify_drop_cam_xy.py` | check-test | 4.6.2, App. E.1.2 | manual (`verify_drop_cam_xy.log`) |
| `check_zero_speed.py` | check-test | 4.6.3 | manual (`check_zero_speed.log`) |
| `test_get_nearby_lane_nodes.py` | check-test | 4.7.3 | manual (unit test) |
| `verify_mapv2_tokens.py` | check-test | 4.7.4 | manual (`verify_mapv2_tokens.log`) |
| `verify_mapv2_off.py` | check-test | 4.7.4 | manual (`verify_mapv2_off.log`) |
| `verify_env_model.py` | check-test | 4.8 | manual (`verify_env_model.log`) |
| `test_env_stage2.py` | check-test | 4.8 | manual (unit test) |
| `verify_rsu_off.py` | check-test | 4.9 | manual (`verify_rsu_off.log`) |
| `verify_env_targets.py` | check-test | 5.3 | manual (`verify_env_targets.log`) |
| `verify_image_cache.py` | check-test | 5.3 | manual (`verify_image_cache.log`) |
| `verify_rsu_image_cache.py` | check-test | 5.3 | manual (`verify_rsu_image_cache.log`) |
| `verify_v2x_ablation.py` | check-test | 6.2.1, App. E.2.3 | manual |
| `verify_relaxed_wta.py` | check-test | App. E.1.2 (configuration E) | manual (`verify_relaxed_wta.log`) |
| `verify_history.py` | check-test | App. E.2.1 | manual (`verify_history.log`) |
| `verify_loss.py` | check-test | App. E.2.2 | `run_wta_soft_seeds.sh` (`verify_loss.log`) |
| `verify_degrade.py` | check-test | App. E.2.4 | manual (`verify_degrade.log`) |
| `check_offroad_gt.py` | diagnostic | 4.6.1, 6.1 (Table 6.2) | manual (`check_offroad_gt.log`; cited in `DECISION_RULES_CORRECTED.md`) |
| `diag_rsu_sensitivity.py` | diagnostic | 6.4.4 (Table 6.12) | manual (`diag_rsu_E_V2XRSU_seed*.log`) |
| `diag_speed_sensitivity.py` | diagnostic | App. E.1.3 (Table E.3) | manual (`diag_speed_sensitivity.log`) |
| `diag_history_sensitivity.py` | diagnostic | App. E.2.1 | manual (cited in `DECISION_RULES_HISTORY.md`) |
| `diag_map_sensitivity.py` | diagnostic | App. E.2.5 | manual (`diag_map_D_seed*.log`) |
| `diag_lidar_behind.py` | figures-tables | 4.6.3 (Fig. 4.2), 6.1 | `make_bev_figures.sh` (`viz_env/thesis_figures/lidar_blindspot.pdf`) |
| `render_env_bev.py` | figures-tables | 6.6, App. D | `make_bev_figures.sh` |
| `make_thesis_tables.py` | figures-tables | 6.2, 6.3 | manual (writes `viz_env/thesis_tables/`) |
| `export_carla_replay.py` | figures-tables | 5.5, App. D (Fig. D.5) | `make_carla_replay.sh` |
| `carla_replay.py` | figures-tables | 5.5, App. D (Fig. D.5) | `make_carla_replay.sh`; imported by `export_carla_replay.py`, `overlay_carla_replay.py` |
| `overlay_carla_replay.py` | figures-tables | 5.5, App. D (Fig. D.5) | `make_carla_replay.sh` |
| `evaluate.py` | legacy-AppG | App. G.1 | manual (pilot) |
| `eval_metrics.py` | legacy-AppG | App. G.1 | manual (pilot) |
| `eval_ablation.py` | legacy-AppG | App. G.1 | manual (pilot) |
| `eval_by_town.py` | legacy-AppG | App. G.1 | manual (pilot) |
| `train_ablation.py` | legacy-AppG | App. G.1 | manual (pilot; superseded by `train.py --no-map`) |
| `diag_scores.py` | legacy-AppG | App. G.1 | manual (`diag_scores_E.log`, `diag_scores_full.log`) |
| `analyze_seeds_test.py` | legacy-AppG | App. G.2 | manual (`analyze_seeds_test.log`) |
| `grid_search_transform.py` | legacy-AppG | App. G.3 | manual; imported by `refine_town03_theta.py`, `refine_town04_theta.py`, `check_town0*_residuals*.py`, `check_residual_distribution.py` |
| `refine_town03_theta.py` | legacy-AppG | App. G.3 | manual |
| `refine_town04_theta.py` | legacy-AppG | App. G.3 | manual |
| `refit_transform.py` | legacy-AppG | App. G.3 | manual (`refit.log`) |
| `refit_town04_constrained.py` | legacy-AppG | App. G.3 | manual |
| `score_transforms.py` | legacy-AppG | App. G.3 | manual |
| `check_town03_residuals.py` | legacy-AppG | App. G.3 | manual |
| `check_town04_residuals.py` | legacy-AppG | App. G.3 | manual |
| `check_town04_residuals_v2.py` | legacy-AppG | App. G.3 | manual |
| `check_residual_distribution.py` | legacy-AppG | App. G.3 | manual |
| `check_heading_alignment.py` | legacy-AppG | App. G.3 | manual |
| `check_heading_raw.py` | legacy-AppG | App. G.3 | manual |
| `plot_alignment.py` | legacy-AppG | App. G.3 | manual (writes `viz_output/align_<town>.png`) |
| `test_transforms.py` | legacy-AppG | App. G.3 | manual (live CARLA, not a unit test) |
| `test_transforms_v2.py` | legacy-AppG | App. G.3 | manual (live CARLA, not a unit test) |
| `four_stream_fusion.py` | legacy | none: early prototype, only used by `trajectory_loss.py`'s `__main__` demo; no thesis result | `trajectory_loss.py` (`__main__` only) |
