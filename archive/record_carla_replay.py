"""
record_carla_replay.py
Records one prediction sample as a video on the CARLA 3D map.

Pipeline: pick a sample -> run the model -> convert to CARLA coordinates ->
spawn a car -> aerial camera sensor -> draw ground truth + predicted modes
with OpenCV -> encode MP4 with ffmpeg.

Usage (carla-venv, CARLA server already running in another tmux pane):
    python record_carla_replay.py Town04
    python record_carla_replay.py Town03
    python record_carla_replay.py Town05

Notes:
  - Sample selection uses only the map and the ground truth, never the
    model's prediction quality (no cherry-picking). Candidates must sit on a
    driving lane and move along that lane's direction; samples near a
    junction are preferred.
  - world.debug.draw_line() is not used: it is not composited into camera
    sensor captures under -RenderOffScreen.
  - One town per run. A 3rd+ load_world() in one process can segfault CARLA
    (Signal 11), so restart the server between towns.
"""
import os
import sys
import glob
import queue
import subprocess
import numpy as np
import cv2
import carla

from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from ground_truth_extractor import get_future_trajectory, HORIZON
from coord_transform import TOWN_TRANSFORMS
from visualize_predictions import (
    load_model, run_single_sample, reconstruct_world_traj, get_agent_world_pos,
    FULL_MODEL_CKPT, MODE_COLORS,
)

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = sys.argv[1] if len(sys.argv) > 1 else "Town04"
OUT_DIR = f"carla_replay_frames_{TOWN}"
OUT_VIDEO = f"viz_output/carla_replay_{TOWN}.mp4"

IMG_W, IMG_H, FOV = 1280, 720, 90
CAM_HEIGHT = 50.0
FPS = 10
REVEAL_SECONDS = 4.0    # trajectories draw progressively over this long
HOLD_SECONDS = 2.0      # then hold the complete picture
CONF_THRESHOLD = 0.05   # modes below this confidence are not drawn

# sample selection
MIN_DISPLACEMENT = 8.0          # meters of net ground-truth motion
MAX_DISPLACEMENT = 25.0
MAX_SCAN = 300                  # max candidates checked against the map
MAX_LANE_DIST = 1.5             # meters from agent to nearest driving lane
MAX_HEADING_DIFF = 35.0         # degrees between GT motion and lane direction
HEADING_REF_DIST = 2.0          # meters of motion used to measure GT heading
JUNCTION_LOOKAHEAD = (10, 20, 30)  # meters ahead to look for a junction


# ----------------------------------------------------------------- helpers
def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # cv2 uses BGR


def put_label(img, text, org, color):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def gt_heading_deg(gt_world, min_dist=HEADING_REF_DIST):
    """Direction of travel in CARLA yaw degrees, from the first ground-truth
    point to the first point at least min_dist away. None if it barely moves."""
    valid = gt_world[~np.isnan(gt_world[:, 0])]
    if len(valid) < 2:
        return None
    d = np.linalg.norm(valid[:, :2] - valid[0, :2], axis=1)
    far = np.nonzero(d >= min_dist)[0]
    if len(far) == 0:
        return None
    v = valid[far[0], :2] - valid[0, :2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def near_junction(wp):
    if wp.is_junction:
        return True
    for d in JUNCTION_LOOKAHEAD:
        if any(w.is_junction for w in wp.next(d)):
            return True
    return False


# ------------------------------------------------------- sample selection
def pick_sample(dataset, loader, model, carla_map, town):
    # 1) cheap prefilter on ground-truth displacement only
    cands = []
    for i, (agent_id, scene, frame, t) in enumerate(dataset.samples):
        if t != town:
            continue
        gt_traj, gt_mask = get_future_trajectory(loader, agent_id, scene, frame, HORIZON)
        if gt_mask.sum() < HORIZON:
            continue
        d = (gt_traj[-1] - gt_traj[0]).norm().item()
        if MIN_DISPLACEMENT <= d <= MAX_DISPLACEMENT:
            cands.append(i)
    print(f"{len(cands)} {town} samples with GT displacement in "
          f"[{MIN_DISPLACEMENT}, {MAX_DISPLACEMENT}] m")
    if not cands:
        raise ValueError(f"No {town} sample with enough motion")

    stride = max(1, len(cands) // MAX_SCAN)
    cands = cands[::stride][:MAX_SCAN]

    # 2) check each candidate against the CARLA map
    rejected = {"off_lane": 0, "no_heading": 0, "heading_mismatch": 0}
    fallback = None
    for i in cands:
        trajs, conf, gt_traj, gt_mask, meta = run_single_sample(model, dataset, i)
        t_name = meta["town"]
        gt_world = reconstruct_world_traj(loader, meta["agent_id"], meta["scene"],
                                          meta["frame"], gt_traj, gt_mask, t_name)
        agent_pos = get_agent_world_pos(loader, meta["agent_id"], meta["scene"],
                                        meta["frame"], t_name)

        loc = carla.Location(float(agent_pos[0]), float(agent_pos[1]), 0.5)
        wp = carla_map.get_waypoint(loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None or loc.distance(wp.transform.location) > MAX_LANE_DIST:
            rejected["off_lane"] += 1
            continue

        yaw = gt_heading_deg(gt_world)
        if yaw is None:
            rejected["no_heading"] += 1
            continue
        if abs(angle_diff(yaw, wp.transform.rotation.yaw)) > MAX_HEADING_DIFF:
            rejected["heading_mismatch"] += 1
            continue

        result = dict(idx=i, trajs=trajs, conf=conf, gt_world=gt_world, gt_mask=gt_mask,
                      agent_pos=agent_pos, yaw=yaw, meta=meta,
                      lane_dist=loc.distance(wp.transform.location),
                      junction=near_junction(wp))
        if result["junction"]:
            print("Picked a sample near a junction.")
            return result
        if fallback is None:
            fallback = result

    if fallback is not None:
        print("No junction sample found; using a lane-aligned sample away from junctions.")
        return fallback
    raise ValueError(f"No usable {town} sample. Rejections: {rejected}")


# ---------------------------------------------------------- projection
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def world_to_pixel(x, y, z, K, world_2_camera):
    """CARLA world -> camera -> image (pinhole). UE4 axes (x fwd, y right,
    z up) are remapped to camera axes (x right, y down, z fwd) before K."""
    point = np.array([x, y, z, 1.0])
    pc = np.dot(world_2_camera, point)
    pc = [pc[1], -pc[2], pc[0]]
    if pc[2] <= 0.01:
        return None
    pi = np.dot(K, pc)
    px, py = pi[0] / pi[2], pi[1] / pi[2]
    if 0 <= px < IMG_W and 0 <= py < IMG_H:
        return int(px), int(py)
    return None


def overlay_frame(frame_bgr, t_visible, gt_world, trajs_world, conf,
                  agent_pos_px, K, world_2_camera):
    img = frame_bgr.copy()

    if agent_pos_px is not None:
        cv2.drawMarker(img, agent_pos_px, (255, 255, 255),
                       markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)

    def project_path(path_world, t):
        pts = []
        for i in range(min(t, path_world.shape[0])):
            if np.isnan(path_world[i, 0]):
                continue
            px = world_to_pixel(path_world[i, 0], path_world[i, 1], 0.3, K, world_2_camera)
            if px is not None:
                pts.append(px)
        return pts

    gt_pts = project_path(gt_world, t_visible)
    for i in range(len(gt_pts) - 1):
        cv2.line(img, gt_pts[i], gt_pts[i + 1], (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    for k in range(trajs_world.shape[0]):
        if conf[k].item() < CONF_THRESHOLD:
            continue
        pts = project_path(trajs_world[k], t_visible)
        color = hex_to_bgr(MODE_COLORS[k % len(MODE_COLORS)])
        thickness = max(2, int(2 + 4 * conf[k].item()))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, thickness=thickness, lineType=cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 6, color, thickness=-1, lineType=cv2.LINE_AA)

    # legend on a semi-transparent dark box
    n_rows = 1 + sum(1 for k in range(trajs_world.shape[0])
                     if conf[k].item() >= CONF_THRESHOLD)
    box = img.copy()
    cv2.rectangle(box, (10, 10), (280, 22 + 28 * n_rows), (25, 25, 25), -1)
    img = cv2.addWeighted(box, 0.65, img, 0.35, 0)

    x0, y0 = 20, 40
    put_label(img, "Ground truth (black)", (x0, y0), (255, 255, 255))
    y = y0
    for k in range(trajs_world.shape[0]):
        if conf[k].item() < CONF_THRESHOLD:
            continue
        y += 28
        put_label(img, f"Mode {k} (conf {conf[k]:.2f})", (x0, y),
                  hex_to_bgr(MODE_COLORS[k % len(MODE_COLORS)]))
    return img


# --------------------------------------------------------------- spawning
def spawn_vehicle(world, vehicle_bp, agent_pos, yaw_deg):
    """Retry at several heights so a curb or prop does not abort the run."""
    for dz in (0.5, 1.0, 2.0, 3.0):
        vehicle = world.try_spawn_actor(
            vehicle_bp,
            carla.Transform(
                carla.Location(x=float(agent_pos[0]), y=float(agent_pos[1]), z=dz),
                carla.Rotation(yaw=float(yaw_deg)),
            ),
        )
        if vehicle is not None:
            print(f"Spawned at z={dz}")
            vehicle.set_simulate_physics(False)  # no dropping or drifting
            return vehicle
    raise RuntimeError("Spawn blocked at every height: position is inside map geometry")


# ------------------------------------------------------------------- main
def main():
    if TOWN not in TOWN_TRANSFORMS:
        raise SystemExit(f"Unknown town '{TOWN}'. Choose from {list(TOWN_TRANSFORMS)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("viz_output", exist_ok=True)
    for old in glob.glob(f"{OUT_DIR}/frame_*.png"):
        os.remove(old)  # never mix frames from an earlier run

    print(f"Target town: {TOWN}")
    print("Loading dataset and model...")
    dataset = PilotDataset()
    loader = V2XMetadataLoader()
    model = load_model(FULL_MODEL_CKPT, use_map=True)

    print(f"Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT} ...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    print(f"Loading {TOWN}...")
    world = client.load_world(TOWN)

    print("Selecting a sample against the CARLA map...")
    sel = pick_sample(dataset, loader, model, world.get_map(), TOWN)
    meta, conf, agent_pos = sel["meta"], sel["conf"], sel["agent_pos"]
    gt_world = sel["gt_world"]
    town = meta["town"]
    print(f"Sample: {meta}")
    print(f"Lane distance {sel['lane_dist']:.2f} m | near junction: {sel['junction']} "
          f"| heading {sel['yaw']:.0f} deg")
    print(f"Mode confidences: {[round(c.item(), 3) for c in conf]}")

    trajs_world = np.stack([
        reconstruct_world_traj(loader, meta["agent_id"], meta["scene"], meta["frame"],
                               sel["trajs"][k], sel["gt_mask"], town)
        for k in range(sel["trajs"].shape[0])
    ])

    settings = world.get_settings()
    original_sync, original_delta = settings.synchronous_mode, settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    actors_to_destroy = []
    total_frames = int((REVEAL_SECONDS + HOLD_SECONDS) * FPS)
    K = build_projection_matrix(IMG_W, IMG_H, FOV)

    try:
        bp_lib = world.get_blueprint_library()
        vehicle = spawn_vehicle(world, bp_lib.filter("vehicle.tesla.model3")[0],
                                agent_pos, sel["yaw"])
        actors_to_destroy.append(vehicle)
        world.tick()

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=float(agent_pos[0]), y=float(agent_pos[1]), z=CAM_HEIGHT),
                carla.Rotation(pitch=-90.0),
            ),
        )
        actors_to_destroy.append(camera)
        world.tick()

        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        agent_pos_px = world_to_pixel(agent_pos[0], agent_pos[1], 0.5, K, world_2_camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        print(f"Capturing {total_frames} raw frames at {FPS} fps...")
        for i in range(total_frames):
            world.tick()
            image = image_queue.get(timeout=10.0)
            image.save_to_disk(f"{OUT_DIR}/frame_{i:05d}.png")
        camera.stop()

    finally:
        print("Cleaning up actors and restoring world settings...")
        for actor in actors_to_destroy:
            actor.destroy()
        settings.synchronous_mode = original_sync
        settings.fixed_delta_seconds = original_delta
        world.apply_settings(settings)

    frame_paths = sorted(glob.glob(f"{OUT_DIR}/frame_*.png"))
    print(f"Captured {len(frame_paths)} raw frames (expected {total_frames}). "
          f"Overlaying trajectories...")

    T = trajs_world.shape[1]
    reveal_frames = int(REVEAL_SECONDS * FPS)
    n_overlaid, n_skipped = 0, 0
    for i, path in enumerate(frame_paths):
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            print(f"  WARNING: skipping unreadable frame {path}")
            n_skipped += 1
            continue
        t_visible = (min(int((i + 1) / max(reveal_frames, 1) * T) + 1, T)
                     if i < reveal_frames else T)
        cv2.imwrite(path, overlay_frame(frame_bgr, t_visible, gt_world, trajs_world,
                                        conf, agent_pos_px, K, world_2_camera))
        n_overlaid += 1
    print(f"Overlaid {n_overlaid} frames ({n_skipped} skipped/unreadable).")

    if n_overlaid == 0:
        print("ERROR: no valid frames to encode, aborting before ffmpeg.")
        return

    print("Encoding to MP4 with ffmpeg...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", f"{OUT_DIR}/frame_%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        OUT_VIDEO,
    ], check=True)
    print(f"Saved {OUT_VIDEO}")


if __name__ == "__main__":
    main()