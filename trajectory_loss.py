"""
trajectory_loss.py
Winner-takes-all loss for multimodal trajectory prediction.

Steps:
1. Compute masked ADE (average displacement error) between each of the 6
   predicted modes and ground truth -> pick the closest mode ("winner").
2. Regression loss: smooth L1 between the winning mode and ground truth,
   masked to only count valid future steps.
3. Classification loss: cross-entropy pushing the confidence head to predict
   the winning mode as the correct one.
4. Total = regression + classification (weighted).

Run in gpu-test-venv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WinnerTakesAllLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight

    def forward(self, trajs: torch.Tensor, conf: torch.Tensor,
                gt_traj: torch.Tensor, gt_mask: torch.Tensor):
        """
        trajs:   [B, n_modes, horizon, 2] predicted candidate trajectories
        conf:    [B, n_modes] predicted confidence (softmax already applied upstream)
        gt_traj: [B, horizon, 2] ground truth ego-centric trajectory
        gt_mask: [B, horizon] bool, True = valid step

        Returns: scalar total loss, plus a dict of components for logging.
        """
        B, n_modes, horizon, _ = trajs.shape

        gt_expanded = gt_traj.unsqueeze(1)          # [B, 1, horizon, 2]
        mask_expanded = gt_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, horizon, 1]

        # per-step squared error, masked, per mode
        sq_err = (trajs - gt_expanded) ** 2                  # [B, n_modes, horizon, 2]
        sq_err = sq_err * mask_expanded                       # zero out invalid steps
        per_step_dist = sq_err.sum(dim=-1).sqrt()              # [B, n_modes, horizon] -- displacement per step

        valid_counts = gt_mask.sum(dim=-1).clamp(min=1).unsqueeze(1)  # [B, 1] avoid div-by-zero
        ade_per_mode = per_step_dist.sum(dim=-1) / valid_counts        # [B, n_modes]

        winner_idx = ade_per_mode.argmin(dim=-1)  # [B] -- index of closest mode per sample

        # ---- regression loss: smooth L1 on the winning mode only ----
        batch_idx = torch.arange(B)
        winning_traj = trajs[batch_idx, winner_idx]  # [B, horizon, 2]

        reg_loss_per_step = F.smooth_l1_loss(winning_traj, gt_traj, reduction="none")  # [B, horizon, 2]
        reg_loss_per_step = reg_loss_per_step * gt_mask.unsqueeze(-1)  # mask invalid steps
        reg_loss = reg_loss_per_step.sum() / (gt_mask.sum() * 2).clamp(min=1)  # normalize by valid (step, xy) count

        # ---- classification loss: push confidence toward the winning mode ----
        cls_loss = F.cross_entropy(conf, winner_idx)  # conf here should be pre-softmax logits ideally,
        # but since conf is already softmaxed upstream (TrajectoryHead applies softmax), we use NLL on log instead:
        # (see note in __main__ / TrajectoryHead about this)

        total_loss = reg_loss + self.cls_weight * cls_loss

        return total_loss, {
            "reg_loss": reg_loss.item(),
            "cls_loss": cls_loss.item(),
            "winner_idx": winner_idx.tolist(),
            "ade_per_mode": ade_per_mode.detach(),
        }


RELAXED_WTA_EPS = 0.05  # fixed for run E; not a CLI knob


class RelaxedWTALoss(nn.Module):
    """Relaxed winner-takes-all: same winner selection, smooth-L1, masking and
    classification term as WinnerTakesAllLoss, but regression is spread over
    all modes -- winner weighted (1-eps), each other mode eps/(K-1). Keeps
    non-winning modes from drifting arbitrarily far from plausible futures.

    reg = (1-eps) * reg_winner + eps/(K-1) * reg_others, where reg_winner is
    computed exactly as in WinnerTakesAllLoss and reg_others is the sum of the
    other modes' masked smooth-L1, normalized by the same (step, xy) count. So
    eps=0 gives 1.0*reg_winner + 0.0*reg_others == WinnerTakesAllLoss exactly.
    """
    def __init__(self, eps: float = RELAXED_WTA_EPS, cls_weight: float = 1.0):
        super().__init__()
        assert 0.0 <= eps < 1.0, f"eps must be in [0, 1), got {eps}"
        self.eps = eps
        self.cls_weight = cls_weight

    def forward(self, trajs: torch.Tensor, conf: torch.Tensor,
                gt_traj: torch.Tensor, gt_mask: torch.Tensor):
        """Same inputs/outputs as WinnerTakesAllLoss.forward."""
        B, n_modes, horizon, _ = trajs.shape
        assert n_modes > 1, "RelaxedWTALoss needs more than one mode"

        # ---- winner selection: identical to WinnerTakesAllLoss ----
        gt_expanded = gt_traj.unsqueeze(1)                    # [B, 1, horizon, 2]
        mask_expanded = gt_mask.unsqueeze(1).unsqueeze(-1)    # [B, 1, horizon, 1]
        sq_err = (trajs - gt_expanded) ** 2
        sq_err = sq_err * mask_expanded
        per_step_dist = sq_err.sum(dim=-1).sqrt()              # [B, n_modes, horizon]
        valid_counts = gt_mask.sum(dim=-1).clamp(min=1).unsqueeze(1)
        ade_per_mode = per_step_dist.sum(dim=-1) / valid_counts  # [B, n_modes]
        winner_idx = ade_per_mode.argmin(dim=-1)               # [B]

        norm = (gt_mask.sum() * 2).clamp(min=1)                # valid (step, xy) count, as in WTA

        # ---- winner regression: identical to WinnerTakesAllLoss ----
        batch_idx = torch.arange(B)
        winning_traj = trajs[batch_idx, winner_idx]
        reg_winner_per_step = F.smooth_l1_loss(winning_traj, gt_traj, reduction="none")
        reg_winner_per_step = reg_winner_per_step * gt_mask.unsqueeze(-1)
        reg_winner = reg_winner_per_step.sum() / norm

        # ---- other modes: same smooth-L1 and masking, winner excluded ----
        reg_all_per_step = F.smooth_l1_loss(trajs, gt_expanded.expand_as(trajs), reduction="none")
        reg_all_per_step = reg_all_per_step * mask_expanded    # [B, n_modes, horizon, 2]
        not_winner = torch.ones(B, n_modes, device=trajs.device)
        not_winner[batch_idx, winner_idx] = 0.0
        reg_others = (reg_all_per_step * not_winner[:, :, None, None]).sum() / norm

        reg_loss = (1.0 - self.eps) * reg_winner + (self.eps / (n_modes - 1)) * reg_others

        # ---- classification: unchanged from WinnerTakesAllLoss ----
        cls_loss = F.cross_entropy(conf, winner_idx)

        total_loss = reg_loss + self.cls_weight * cls_loss

        return total_loss, {
            "reg_loss": reg_loss.item(),
            "cls_loss": cls_loss.item(),
            "reg_winner": reg_winner.item(),
            "reg_others": reg_others.item(),
            "winner_idx": winner_idx.tolist(),
            "ade_per_mode": ade_per_mode.detach(),
        }


def build_loss(name: str):
    """'wta' -> WinnerTakesAllLoss (default); 'relaxed_wta' -> RelaxedWTALoss(eps=RELAXED_WTA_EPS)."""
    if name == "wta":
        return WinnerTakesAllLoss()
    if name == "relaxed_wta":
        return RelaxedWTALoss(eps=RELAXED_WTA_EPS)
    raise ValueError(f"unknown loss {name!r}")


if __name__ == "__main__":
    from ego_stream import EgoStream
    from ego_sensor_encoder import load_semantic_lidar
    from ego_camera_encoder import load_image
    from coop_sensor_stream import CoopSensorStream
    from find_coop_neighbors import find_coop_neighbors
    from v2x_message_stream import V2XMetadataLoader, V2XMessageStream
    from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes
    from map_cross_attention_fusion import MapCrossAttentionFusion, gather_nearby_lane_embeddings
    from coord_transform import v2xsim_to_carla
    from four_stream_fusion import FourStreamFusion, AGENT_ID, SCENE, FRAME, TOWN, EMBED_DIM
    from ground_truth_extractor import get_future_trajectory

    # ---------- rebuild the four embeddings (same as four_stream_fusion.py) ----------
    lidar_path = f"/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_{AGENT_ID}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
    points = load_semantic_lidar(lidar_path)

    cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_" + str(AGENT_ID) + "/scene_" + str(SCENE) + f"_{FRAME:06d}.jpg"
    views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    images = [load_image(cam_base.format(view=v)) for v in views]

    ego_encoder = EgoStream()
    ego_embed = ego_encoder(points, images).unsqueeze(0)

    loader = V2XMetadataLoader()

    cam, _ = loader.get_cam(AGENT_ID, SCENE, FRAME)
    cpm_objects = loader.get_cpm(AGENT_ID, SCENE, FRAME)
    v2x_encoder = V2XMessageStream()
    v2x_embed = v2x_encoder(cam, cpm_objects).unsqueeze(0)

    neighbor_ids = find_coop_neighbors(loader, ego_agent_id=AGENT_ID, scene=SCENE, frame=FRAME)
    coop_points, coop_images = [], []
    for nid in neighbor_ids:
        n_lidar_path = f"/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_{nid}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
        coop_points.append(load_semantic_lidar(n_lidar_path))
        n_cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_" + str(nid) + "/scene_" + str(SCENE) + f"_{FRAME:06d}.jpg"
        coop_images.append([load_image(n_cam_base.format(view=v)) for v in views])
    coop_encoder = CoopSensorStream()
    coop_embed = coop_encoder(coop_points, coop_images).unsqueeze(0)

    data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")
    map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)
    lane_embeds = map_encoder(data.x, data.edge_index, data.edge_type)
    ego_x, ego_y = cam[0].item(), cam[1].item()
    carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y)
    agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32)
    agent_pos_norm = (agent_pos - mean) / std
    k = 5
    nearby_idx = get_nearby_lane_nodes(agent_pos_norm, data.x, k=k)
    nearby_embeds = gather_nearby_lane_embeddings(lane_embeds, nearby_idx).unsqueeze(0)
    map_fusion = MapCrossAttentionFusion(agent_dim=EMBED_DIM, lane_dim=128, d_model=128)
    map_embed = map_fusion(ego_embed, nearby_embeds)

    # ---------- fusion + trajectory head ----------
    fusion = FourStreamFusion()
    fused, trajs, conf = fusion(ego_embed, coop_embed, v2x_embed, map_embed)

    # ---------- ground truth ----------
    gt_traj, gt_mask = get_future_trajectory(loader, agent_id=AGENT_ID, scene=SCENE, frame=FRAME)
    gt_traj = gt_traj.unsqueeze(0)   # [1, 30, 2]
    gt_mask = gt_mask.unsqueeze(0)   # [1, 30]

    # ---------- loss ----------
    loss_fn = WinnerTakesAllLoss()
    loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Reg loss:   {info['reg_loss']:.4f}")
    print(f"Cls loss:   {info['cls_loss']:.4f}")
    print(f"Winner mode index: {info['winner_idx']}")
    print(f"ADE per mode: {info['ade_per_mode']}")