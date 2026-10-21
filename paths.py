"""
paths.py
Locations of the dataset, the caches and CARLA, in one place. Each root can be overridden by an environment
variable; the defaults are the paths of the thesis machine (HOME=/home/karra), so with no variable set every path
below is the same string the scripts used before paths.py.

  V2X_SIM_ROOT  V2X-Sim 2.0 (v2.0/ metadata JSON, sweeps/)          default ~/V2X-Sim-2.0
  HD_CACHE      caches: images/, images_rsu/, env_targets/           default ~/hd_cache
  CARLA_ROOT    CARLA 0.9.15 install (CarlaUE4.sh, OpenDRIVE maps)   default ~/carla_sim

Example on another machine:  V2X_SIM_ROOT=/data/V2X-Sim-2.0 HD_CACHE=/scratch/hd_cache python train.py ...
"""
import os


def _root(var, default):
    return os.path.expanduser(os.environ.get(var) or default)


V2X_SIM_ROOT = _root("V2X_SIM_ROOT", "~/V2X-Sim-2.0")
HD_CACHE = _root("HD_CACHE", "~/hd_cache")
CARLA_ROOT = _root("CARLA_ROOT", "~/carla_sim")

V2X_META_DIR = os.path.join(V2X_SIM_ROOT, "v2.0")            # sample_data.json, ego_pose.json, ...
V2X_SWEEPS_DIR = os.path.join(V2X_SIM_ROOT, "sweeps")        # LIDAR_TOP_id_<a>/, CAM_FRONT_id_<a>/, ...
IMAGE_CACHE_DIR = os.path.join(HD_CACHE, "images")           # build_image_cache.py
RSU_IMAGE_CACHE_DIR = os.path.join(HD_CACHE, "images_rsu")   # build_rsu_image_cache.py
ENV_CACHE_DIR = os.path.join(HD_CACHE, "env_targets")        # build_env_target_cache.py
ENV_TARGETS_NPZ = os.path.join(ENV_CACHE_DIR, "env_targets.npz")
CARLA_XODR_DIR = os.path.join(CARLA_ROOT, "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive")
