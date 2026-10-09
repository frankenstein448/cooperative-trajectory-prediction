"""
motion_utils.py
Shared constant-velocity / displacement helpers, used by eval_scenes.py's CV
baseline and (later) the dataset's speed field fix -- kept in one place so
both stay on the exact same convention.
"""
import math

from v2x_message_stream import quaternion_to_yaw


def get_cv_velocity(loader, agent_id, scene, frame):
    """Displacement frame-1 -> frame, in metres, rotated into the heading at
    `frame` (same convention as ground_truth_extractor.get_future_trajectory).
    No timestamps/speed field involved -- this is metres per frame-step.
    Returns (ok, vx, vy); ok=False (vx=vy=0.0) if frame-1 has no data for
    this agent/scene.
    """
    cur_rec = loader.find_sample_data(agent_id, scene, frame)
    cur_pose = loader.ego_pose[cur_rec["ego_pose_token"]]
    cur_x, cur_y, _ = cur_pose["translation"]
    cur_heading = quaternion_to_yaw(cur_pose["rotation"])
    cos_h, sin_h = math.cos(-cur_heading), math.sin(-cur_heading)

    try:
        prev_rec = loader.find_sample_data(agent_id, scene, frame - 1)
    except ValueError:
        return False, 0.0, 0.0

    prev_pose = loader.ego_pose[prev_rec["ego_pose_token"]]
    px, py, _ = prev_pose["translation"]
    dx, dy = cur_x - px, cur_y - py
    local_x = dx * cos_h - dy * sin_h
    local_y = dx * sin_h + dy * cos_h
    return True, local_x, local_y
