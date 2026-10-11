"""
visualize_predictions.py
Renders predicted trajectories (6 modes, colored by confidence, time-graded
along each path) against ground truth and the local HD map lane graph, in
real-world (CARLA) meters.

Produces:
  1. Per-town static PNG, full model -- for report figures.
  2. Full-model vs no-map ablation side-by-side PNG, same sample -- the
     most direct visual evidence for the ablation's ~13% degradation.
  3. Animated MP4 per town, predicted paths drawn progressively across the
     30-step horizon -- for playing during the defense.

Coordinate handling (the part that's easy to get wrong):
  - gt_traj / model trajs are EGO-CENTRIC: (0,0) = agent's position at the
    current frame, axes rotated so "forward" = the agent's current heading
    (see ground_truth_extractor.get_future_trajectory). NOT world/CARLA
    coordinates directly.
  - Each relative point is rotated back into V2X-Sim WORLD coordinates
    using the agent's current world heading + position, then passed
    through coord_transform.v2xsim_to_carla() -- reusing the same
    validated per-town transform used everywhere else in this project,
    rather than re-deriving a CARLA-frame heading offset by hand.
  - Lane graph node positions (data.x from load_graph) are NORMALIZED.
    Denormalized back to real CARLA meters via load_graph's own mean/std
    before plotting.

Requires ffmpeg on PATH for the MP4 output (check: `ffmpeg -version`).
Run in gpu-test-venv.
"""
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from rgcn_map_encoder import load_graph
from coord_transform import v2xsim_to_carla
from v2x_message_stream import V2XMetadataLoader, quaternion_to_yaw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_MODEL_CKPT = "checkpoints_e2e/best_model.pt"
NO_MAP_CKPT = "checkpoints_e2e_no_map/best_model.pt"
OUT_DIR = "viz_output"

MODE_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
GT_COLOR = "#000000"


def reconstruct_world_traj(loader, agent_id, scene, frame, local_traj, mask, town):
    """local_traj: [T,2] ego-centric relative offsets -> [T,2] CARLA-frame
    meters. Invalid (mask=False) steps become NaN so they plot as gaps."""
    cur_rec = loader.find_sample_data(agent_id, scene, frame)
    cur_pose = loader.ego_pose[cur_rec["ego_pose_token"]]
    cur_x, cur_y, _ = cur_pose["translation"]
    cur_heading = quaternion_to_yaw(cur_pose["rotation"])
    cos_h, sin_h = math.cos(cur_heading), math.sin(cur_heading)  # inverse of extractor's -heading rotation

    T = local_traj.shape[0]
    world_pts = np.full((T, 2), np.nan, dtype=np.float64)
    for t in range(T):
        if not mask[t]:
            continue
        lx, ly = local_traj[t, 0].item(), local_traj[t, 1].item()
        dx = lx * cos_h - ly * sin_h
        dy = lx * sin_h + ly * cos_h
        wx, wy = cur_x + dx, cur_y + dy
        cx, cy = v2xsim_to_carla(wx, wy, town)
        world_pts[t] = [cx, cy]
    return world_pts


def get_agent_world_pos(loader, agent_id, scene, frame, town):
    cur_rec = loader.find_sample_data(agent_id, scene, frame)
    cur_pose = loader.ego_pose[cur_rec["ego_pose_token"]]
    cur_x, cur_y, _ = cur_pose["translation"]
    return v2xsim_to_carla(cur_x, cur_y, town)


def load_model(ckpt_path, use_map, history_frames=0, history_encoder="gru"):
    model = EndToEndFusion(device=DEVICE, use_map=use_map, history_frames=history_frames,
                           history_encoder=history_encoder).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_single_sample(model, dataset, idx):
    """Runs the model on ONE dataset sample via a batch-of-1."""
    item = dataset[idx]
    batch = pilot_collate_fn([item])
    with torch.no_grad():
        _, trajs, conf_logits = model(batch)
    conf = torch.softmax(conf_logits, dim=-1)[0].cpu()  # [6]
    trajs = trajs[0].cpu()  # [6, 30, 2]
    gt_traj = batch["gt_traj"][0]
    gt_mask = batch["gt_mask"][0]
    meta = {"agent_id": item["agent_id"], "scene": item["scene"], "frame": item["frame"], "town": item["town"]}
    return trajs, conf, gt_traj, gt_mask, meta


def draw_lane_graph(ax, town, graph_cache):
    """Lane centerlines, denormalized to real meters, cached per town."""
    if town not in graph_cache:
        data, id_to_idx, mean, std = load_graph(f"topology/{town}_topology.pkl")
        real_pos = (data.x * std + mean).numpy()
        graph_cache[town] = (real_pos, data.edge_index.numpy())
    real_pos, edge_index = graph_cache[town]
    for i in range(edge_index.shape[1]):
        p0 = real_pos[edge_index[0, i]]
        p1 = real_pos[edge_index[1, i]]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#cccccc", linewidth=1.0, zorder=1)


def draw_trajectories(ax, trajs_world, conf, gt_world, agent_pos, title):
    """trajs_world: [6,T,2] CARLA-frame (may contain NaN gaps). Each mode's
    path is time-graded (light->dark) and thickness-scaled by confidence."""
    ax.scatter([agent_pos[0]], [agent_pos[1]], c="black", marker="*", s=200, zorder=5, label="Agent (t=0)")
    for k in range(trajs_world.shape[0]):
        path = trajs_world[k]
        valid = ~np.isnan(path[:, 0])
        if valid.sum() < 2:
            continue
        pts = path[valid]
        T = pts.shape[0]
        for t in range(T - 1):
            alpha = 0.35 + 0.65 * (t / max(T - 1, 1))
            lw = 1.0 + 3.0 * conf[k].item()
            ax.plot(pts[t:t + 2, 0], pts[t:t + 2, 1], color=MODE_COLORS[k % len(MODE_COLORS)],
                     alpha=alpha, linewidth=lw, zorder=3)
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], color=MODE_COLORS[k % len(MODE_COLORS)],
                    s=30 + 100 * conf[k].item(), zorder=4, edgecolors="white", linewidths=0.5)

    valid_gt = ~np.isnan(gt_world[:, 0])
    if valid_gt.sum() >= 2:
        pts = gt_world[valid_gt]
        ax.plot(pts[:, 0], pts[:, 1], color=GT_COLOR, linewidth=2.5, linestyle="--", zorder=4, label="Ground truth")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("CARLA X (m)")
    ax.set_ylabel("CARLA Y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def zoom_to_content(ax, agent_pos, gt_world, trajs_world, pad=15):
    all_pts = [np.atleast_2d(agent_pos)]
    for arr in [gt_world] + [trajs_world[k] for k in range(trajs_world.shape[0])]:
        valid = ~np.isnan(arr[:, 0])
        if valid.any():
            all_pts.append(arr[valid])
    all_pts = np.vstack(all_pts)
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _get_world(model, dataset, loader, idx):
    trajs, conf, gt_traj, gt_mask, meta = run_single_sample(model, dataset, idx)
    town = meta["town"]
    trajs_world = np.stack([
        reconstruct_world_traj(loader, meta["agent_id"], meta["scene"], meta["frame"], trajs[k], gt_mask, town)
        for k in range(trajs.shape[0])
    ])
    gt_world = reconstruct_world_traj(loader, meta["agent_id"], meta["scene"], meta["frame"], gt_traj, gt_mask, town)
    agent_pos = get_agent_world_pos(loader, meta["agent_id"], meta["scene"], meta["frame"], town)
    return trajs_world, conf, gt_world, agent_pos, meta


def make_static_figure(model, dataset, loader, idx, graph_cache, out_path, model_label=""):
    trajs_world, conf, gt_world, agent_pos, meta = _get_world(model, dataset, loader, idx)
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_lane_graph(ax, meta["town"], graph_cache)
    title = f"{meta['town']} | agent {meta['agent_id']} | scene {meta['scene']} frame {meta['frame']}{model_label}"
    draw_trajectories(ax, trajs_world, conf, gt_world, agent_pos, title)
    zoom_to_content(ax, agent_pos, gt_world, trajs_world)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_comparison_figure(full_model, no_map_model, dataset, loader, idx, graph_cache, out_path):
    """Side-by-side: full model (left) vs no-map ablation (right), SAME sample."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    town_label = None
    for ax, model, label in [(axes[0], full_model, "Full model (with map)"),
                              (axes[1], no_map_model, "No-map ablation")]:
        trajs_world, conf, gt_world, agent_pos, meta = _get_world(model, dataset, loader, idx)
        town_label = meta
        draw_lane_graph(ax, meta["town"], graph_cache)
        draw_trajectories(ax, trajs_world, conf, gt_world, agent_pos, label)
        zoom_to_content(ax, agent_pos, gt_world, trajs_world)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Full model vs no-map ablation -- {town_label['town']}, agent {town_label['agent_id']}, "
                 f"scene {town_label['scene']}, frame {town_label['frame']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_animation(model, dataset, loader, idx, graph_cache, out_path, model_label="", fps=6):
    trajs_world, conf, gt_world, agent_pos, meta = _get_world(model, dataset, loader, idx)
    town = meta["town"]

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_lane_graph(ax, town, graph_cache)
    zoom_to_content(ax, agent_pos, gt_world, trajs_world)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("CARLA X (m)")
    ax.set_ylabel("CARLA Y (m)")
    ax.set_title(f"{town} | agent {meta['agent_id']} | scene {meta['scene']} frame {meta['frame']}{model_label}",
                 fontsize=11)
    ax.scatter([agent_pos[0]], [agent_pos[1]], c="black", marker="*", s=200, zorder=5)

    mode_lines = [ax.plot([], [], color=MODE_COLORS[k % len(MODE_COLORS)],
                           linewidth=1.5 + 3.0 * conf[k].item(), alpha=0.9)[0]
                  for k in range(trajs_world.shape[0])]
    mode_heads = [ax.scatter([], [], color=MODE_COLORS[k % len(MODE_COLORS)], s=60,
                              edgecolors="white", linewidths=0.5, zorder=4)
                  for k in range(trajs_world.shape[0])]
    gt_line, = ax.plot([], [], color=GT_COLOR, linewidth=2.5, linestyle="--", zorder=4)
    gt_head = ax.scatter([], [], color=GT_COLOR, s=60, zorder=5)

    legend_elems = [Line2D([0], [0], color=GT_COLOR, linestyle="--", lw=2.5, label="Ground truth")]
    legend_elems += [Line2D([0], [0], color=MODE_COLORS[k % len(MODE_COLORS)], lw=2,
                             label=f"Mode {k} (conf {conf[k]:.2f})") for k in range(trajs_world.shape[0])]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7)

    T = trajs_world.shape[1]

    def update(frame_i):
        t = frame_i + 1
        for k in range(trajs_world.shape[0]):
            pts = trajs_world[k, :t]
            valid = ~np.isnan(pts[:, 0])
            mode_lines[k].set_data(pts[valid, 0], pts[valid, 1])
            if valid.any():
                mode_heads[k].set_offsets([pts[valid][-1]])
        gt_pts = gt_world[:t]
        valid_gt = ~np.isnan(gt_pts[:, 0])
        gt_line.set_data(gt_pts[valid_gt, 0], gt_pts[valid_gt, 1])
        if valid_gt.any():
            gt_head.set_offsets([gt_pts[valid_gt][-1]])
        return mode_lines + mode_heads + [gt_line, gt_head]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved {out_path}")


def pick_town_representatives(dataset):
    """First val-set-eligible sample per town encountered, for reproducible defaults.
    Edit this dict manually if you want specific (agent_id, scene, frame) samples instead."""
    seen = {}
    for i, (agent_id, scene, frame, town) in enumerate(dataset.samples):
        if town not in seen:
            seen[town] = i
        if len(seen) == 3:
            break
    return seen


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset and models...")
    dataset = PilotDataset()
    loader = V2XMetadataLoader()
    graph_cache = {}

    full_model = load_model(FULL_MODEL_CKPT, use_map=True)
    no_map_model = load_model(NO_MAP_CKPT, use_map=False)

    reps = pick_town_representatives(dataset)
    print(f"Representative sample indices: {reps}")

    print("\n--- Static per-town figures (full model) ---")
    for town, idx in reps.items():
        make_static_figure(full_model, dataset, loader, idx, graph_cache,
                            f"{OUT_DIR}/static_{town}_full.png", model_label=" (full model)")

    print("\n--- Full vs no-map comparison figures ---")
    for town, idx in reps.items():
        make_comparison_figure(full_model, no_map_model, dataset, loader, idx, graph_cache,
                                f"{OUT_DIR}/comparison_{town}.png")

    print("\n--- Animated MP4s (full model) ---")
    for town, idx in reps.items():
        make_animation(full_model, dataset, loader, idx, graph_cache,
                        f"{OUT_DIR}/anim_{town}_full.mp4", model_label=" (full model)")

    print("\nAll visualizations complete. See viz_output/")