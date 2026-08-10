"""
ground_truth_extractor.py
Extracts an agent's real future trajectory (next HORIZON frames), expressed
in the ego-centric frame at the current timestep (relative position, rotated
by current heading). Returns a mask for frames where data is missing.
Run in gpu-test-venv.
"""
import math
import torch

from v2x_message_stream import V2XMetadataLoader, quaternion_to_yaw

HORIZON = 30  # matches TrajectoryHead


def get_future_trajectory(loader: V2XMetadataLoader, agent_id: int, scene: int, frame: int,
                           horizon: int = HORIZON):
    """
    Returns:
        traj: [horizon, 2] ego-centric (x, y) offsets, rotated into current heading.
              Missing frames are zero-filled.
        mask: [horizon] bool, True = valid ground truth at that step.
    """
    # current pose (the "origin" for the ego-centric frame)
    cur_rec = loader.find_sample_data(agent_id, scene, frame)
    cur_pose = loader.ego_pose[cur_rec["ego_pose_token"]]
    cur_x, cur_y, _ = cur_pose["translation"]
    cur_heading = quaternion_to_yaw(cur_pose["rotation"])

    cos_h, sin_h = math.cos(-cur_heading), math.sin(-cur_heading)  # inverse rotation

    traj = torch.zeros((horizon, 2), dtype=torch.float32)
    mask = torch.zeros(horizon, dtype=torch.bool)

    for step in range(1, horizon + 1):
        future_frame = frame + step
        try:
            rec = loader.find_sample_data(agent_id, scene, future_frame)
        except ValueError:
            continue  # agent has no data at this future frame -- leave zero-filled, mask=False

        pose = loader.ego_pose[rec["ego_pose_token"]]
        fx, fy, _ = pose["translation"]

        dx, dy = fx - cur_x, fy - cur_y
        # rotate global displacement into ego-centric frame (undo current heading)
        local_x = dx * cos_h - dy * sin_h
        local_y = dx * sin_h + dy * cos_h

        traj[step - 1] = torch.tensor([local_x, local_y])
        mask[step - 1] = True

    return traj, mask


if __name__ == "__main__":
    loader = V2XMetadataLoader()
    traj, mask = get_future_trajectory(loader, agent_id=4, scene=100, frame=6)

    print(f"Trajectory shape: {traj.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Valid future steps: {mask.sum().item()} / {mask.shape[0]}")
    print(f"First 5 steps:\n{traj[:5]}")
    print(f"Mask (first 10): {mask[:10]}")
    print(f"\nAll 30 steps:\n{traj}")
    total_displacement = (traj[-1] - traj[0]).norm().item()
    print(f"\nNet displacement (start to end of horizon): {total_displacement:.2f}m")