"""
export_carla_replay.py
Stage 0 of the CARLA replay (visualisation only): per frame of one tuning-split scene / ego, a plain .npz (numeric
arrays + a JSON meta string, no pickle) that carla_replay.py (carla-venv) and overlay_carla_replay.py read. Everything
is in V2X-Sim WORLD coordinates (the mirror to CARLA is applied by the readers).

Per frame: every kept target of the env cache (the rows render_env_bev.py draws): tid, xy_world, yaw_world, size
(w, l, h), two_wheeler (cat4 bicycle / motorcycle), cls (0 ego, 1 coop CAV only, 2 RSU only, 3 nobody), moving
(speed > 0), gt_world [N, 30, 2] and gt_mask; the ego / CAV / RSU poses (xy, yaw; absent agents are NaN); per model
known [N], modes_world [N, 6, 30, 2] (NaN where not known) and top1 [N] (-1 where not known).

Checks (printed, asserted <= 1e-3 m): the world ground truth taken back to the ego frame reproduces the cache's
fut_ego and cur_ego; the world modes taken back to the ego frame reproduce render_env_bev.own_to_ego of the cached
predictions; load_preds re-checks minADE6 / ADE1 against the run's eval_env_tuning npz.

Tuning split v2 only, seed 0, epoch 12; reads viz_env/pred (predicts there only if a file is missing).
Run in gpu-test-venv from the data checkout:
  nice -n 19 python ~/hd-maps-dev/export_carla_replay.py --scene 16 --agent 4 --runs E_V2X_seed0,E_EgoOnly_seed0
"""
import os
import sys
import json
import math
import argparse

import numpy as np

from render_env_bev import (TUNING, COOP_CAVS, RSU_AGENT, busy, run_cfg, load_preds, scene_rows, vis_class, to_frame,
                            own_to_ego)


def own_to_world(p, xy, yaw):
    """p [..., 2] in a target's own frame -> world, given its xy_world and yaw_world."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([p[..., 0] * c - p[..., 1] * s + xy[0], p[..., 0] * s + p[..., 1] * c + xy[1]], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--agent", type=int, required=True)
    ap.add_argument("--runs", default="E_V2X_seed0,E_EgoOnly_seed0")
    ap.add_argument("--viz", default="viz_env", help="render_env_bev.py --out (its pred/ cache is read)")
    ap.add_argument("--out", default="viz_env/carla_replay")
    args = ap.parse_args()
    running = busy()
    if running:
        sys.exit("refusing: a run / training / evaluation process is running:\n  " + "\n  ".join(running))
    s, agent = args.scene, args.agent
    if s not in TUNING:
        sys.exit(f"scene {s} is not a tuning scene of split v2 {sorted(TUNING)}; refusing")
    from env_dataset import load_env_cache, known_mask, ENV_CACHE_DIR
    from v2x_message_stream import V2XMetadataLoader, quaternion_to_yaw
    runs = args.runs.split(",")
    cfgs = {r: run_cfg(r) for r in runs}
    cache = load_env_cache()
    fut_ego_ref = np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz"))["fut_ego"]
    rows = scene_rows(cache, s)
    rows = rows[cache["sample_agent"][rows] == agent]
    rows = rows[np.argsort(cache["sample_frame"][rows])]
    assert len(rows), f"no env samples for scene {s}, ego {agent}"
    preds = {r: load_preds(r, s, agent, args.viz, cache) for r in runs}
    L = V2XMetadataLoader()

    def pose(a, f):
        try:
            p = L.ego_pose[L.find_sample_data(a, s, f)["ego_pose_token"]]
        except ValueError:
            return np.full(3, np.nan)
        return np.array([p["translation"][0], p["translation"][1], quaternion_to_yaw(p["rotation"])])

    out_dir = os.path.join(args.out, f"scene{s}_ego{agent}")
    os.makedirs(out_dir, exist_ok=True)
    err = {"gt_fut": 0.0, "gt_cur": 0.0, "modes": 0.0}
    for r in rows:
        f = int(cache["sample_frame"][r])
        ego = pose(agent, f)
        exy, eyaw = ego[:2], ego[2]
        tid = np.arange(int(cache["sample_ptr"][r]), int(cache["sample_ptr"][r + 1]))
        xy, yaw = cache["xy_world"][tid].astype(np.float64), cache["yaw_world"][tid].astype(np.float64)
        n = len(tid)
        gt = np.stack([own_to_world(cache["fut_own"][t].astype(np.float64), xy[j], yaw[j])
                       for j, t in enumerate(tid)]) if n else np.zeros((0, 30, 2))
        if n:
            err["gt_fut"] = max(err["gt_fut"], float(np.abs(to_frame(gt, exy, eyaw) - fut_ego_ref[tid]).max()))
            err["gt_cur"] = max(err["gt_cur"], float(np.abs(to_frame(xy, exy, eyaw) - cache["cur_ego"][tid]).max()))
        cur_ego, yaw_ego = cache["cur_ego"][tid].astype(np.float64), cache["yaw_ego"][tid].astype(np.float64)
        d = {"tid": tid, "xy_world": xy, "yaw_world": yaw, "size": cache["size"][tid].astype(np.float64),
             "two_wheeler": cache["cat4"][tid] >= 2, "cls": vis_class(cache["vis"][tid], cache["rsu_vis"][tid]),
             "moving": cache["speed"][tid] > 0, "gt_world": gt, "gt_mask": cache["fut_mask"][tid],
             "ego_pose": ego, "cav_ids": np.array([a for a in COOP_CAVS if a != agent]),
             "rsu_pose": pose(RSU_AGENT, f)}
        d["cav_poses"] = np.stack([pose(a, f) for a in d["cav_ids"]])
        for run in runs:
            known = known_mask(cache["vis"][tid], cfgs[run]["v2x"],
                               cache["rsu_vis"][tid] if cfgs[run].get("use_rsu", False) else None)
            modes = np.full((n, 6, 30, 2), np.nan)
            top1 = np.full(n, -1)
            for j, t in enumerate(tid):
                if not known[j]:
                    continue
                assert int(t) in preds[run], f"{run}: known target {t} without a prediction in frame {f}"
                trajs, logits = preds[run][int(t)]
                trajs = trajs.astype(np.float64)
                modes[j] = own_to_world(trajs, xy[j], yaw[j])
                top1[j] = int(np.argmax(logits))
                err["modes"] = max(err["modes"], float(np.abs(to_frame(modes[j], exy, eyaw)
                                                              - own_to_ego(trajs, cur_ego[j], yaw_ego[j])).max()))
            base = run.rsplit("_seed", 1)[0]
            d[f"{base}_known"], d[f"{base}_modes_world"], d[f"{base}_top1"] = known, modes, top1
        d["meta"] = json.dumps({"town": TUNING[s], "scene": s, "agent": agent, "frame": f, "split": "tuning",
                                "runs": runs, "models": [r.rsplit("_seed", 1)[0] for r in runs],
                                "coords": "V2X-Sim world (x, y, yaw rad); CARLA = (x, -y, -yaw)"})
        np.savez(os.path.join(out_dir, f"frame_{f:03d}.npz"), **d)
    print(f"{TUNING[s]} scene {s}, ego CAV {agent}: {len(rows)} frames ({cache['sample_frame'][rows[0]]}.."
          f"{cache['sample_frame'][rows[-1]]}) -> {out_dir}")
    print("checks (max |world -> ego frame - reference|, m): "
          + ", ".join(f"{k} {v:.2e}" for k, v in err.items()))
    assert max(err.values()) <= 1e-3, "world conversion does not reproduce the ego-frame values"


if __name__ == "__main__":
    main()
