"""
find_coop_neighbors.py
Real neighbor-finding for the coop stream: given an ego agent + scene/frame,
returns IDs of nearby CAV/RSU agents (LiDAR+camera compatible only).
Run in gpu-test-venv.
"""
import math
from v2x_message_stream import V2XMetadataLoader

COOP_RADIUS_M = 200.0  # V2X comms range -- separate from CPM's 50m object-detection radius
CAMERA_COMPATIBLE_AGENTS = [1, 2, 3]  # CAVs with LiDAR+camera; agent 0 excluded (LiDAR-only)


def find_coop_neighbors(loader: V2XMetadataLoader, ego_agent_id: int, scene: int, frame: int,
                         candidate_ids=CAMERA_COMPATIBLE_AGENTS, radius=COOP_RADIUS_M, verbose=False):
    """Returns list of agent IDs within `radius` meters of ego_agent_id, excluding itself."""
    ego_cam, _ = loader.get_cam(ego_agent_id, scene, frame)
    ego_x, ego_y = ego_cam[0].item(), ego_cam[1].item()

    neighbors = []
    for cand_id in candidate_ids:
        if cand_id == ego_agent_id:
            continue
        try:
            cand_cam, _ = loader.get_cam(cand_id, scene, frame)
        except ValueError:
            if verbose:
                print(f"  agent {cand_id}: no data at this frame")
            continue
        cand_x, cand_y = cand_cam[0].item(), cand_cam[1].item()
        dist = math.hypot(cand_x - ego_x, cand_y - ego_y)
        if verbose:
            print(f"  agent {cand_id}: distance = {dist:.1f}m")
        if dist <= radius:
            neighbors.append((cand_id, dist))

    neighbors.sort(key=lambda t: t[1])
    return [agent_id for agent_id, _ in neighbors]


if __name__ == "__main__":
    loader = V2XMetadataLoader()

    print("=== Single frame (scene 100, frame 6) ===")
    neighbors = find_coop_neighbors(loader, ego_agent_id=4, scene=100, frame=6, verbose=True)
    print(f"Neighbors: {neighbors}\n")

    print("=== Scanning frames 6-40 in scene 100 ===")
    for frame in range(6, 41):
        try:
            neighbors = find_coop_neighbors(loader, ego_agent_id=4, scene=100, frame=frame)
            if neighbors:
                print(f"frame {frame}: neighbors = {neighbors}")
        except ValueError:
            continue  # agent 4 itself has no data at this frame