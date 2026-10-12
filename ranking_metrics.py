"""
ranking_metrics.py
Mode-ranking metrics for eval_scenes.py (evaluation only, nothing here is trained on).

  - Brier-minFDE: minFDE6 + (1 - p)^2, p = softmax mode score of the min-FDE mode
    (the mode chosen independently by FDE, as in fde6).
  - mode diversity: mean pairwise distance between the K modes at the final horizon
    step (the prediction's last step, whatever the GT mask).
  - Laplace calibration (laplace_mixture runs only), for the top-scored mode:
    a Laplace(mu, b) coordinate lies in |x - mu| <= b * ln(1 / (1 - q)) with
    probability q. "xy" coverage: each valid GT coordinate inside its level-L
    interval (nominal L). "pt" coverage: a valid GT point inside the box of both
    coordinates' level-sqrt(L) intervals (nominal L, as x and y are independent
    under the model).
"""
import math
import torch

COVERAGE_LEVELS = (50, 90)  # percent


def brier_min_fde(fde_modes, scores):
    """fde_modes, scores: [B, K] -> [B]."""
    best = fde_modes.argmin(dim=1, keepdim=True)
    p = torch.softmax(scores, dim=-1).gather(1, best).squeeze(1)
    return fde_modes.gather(1, best).squeeze(1) + (1 - p) ** 2


def mode_diversity(trajs):
    """trajs: [B, K, T, 2] -> [B], mean distance over the K(K-1)/2 mode pairs at step T-1."""
    final = trajs[:, :, -1, :]
    K = final.shape[1]
    return torch.cdist(final, final).sum(dim=(1, 2)) / (K * (K - 1))


def laplace_coverage(trajs, scales, gt, mask, top):
    """trajs, scales: [B, K, T, 2]; gt: [B, T, 2]; mask: [B, T]; top: [B] mode index.
    Returns (pt_hits, xy_hits, npts): dicts level -> [B] hit counts over valid steps
    (points / coordinates), and [B] valid-step counts."""
    B = trajs.shape[0]
    idx = torch.arange(B, device=trajs.device)
    err = (gt - trajs[idx, top]).abs() / scales[idx, top]   # [B, T, 2], |x - mu| / b
    m = mask.float()
    pt_hits, xy_hits = {}, {}
    for lvl in COVERAGE_LEVELS:
        q = lvl / 100
        xy_in = err <= math.log(1 / (1 - q))
        xy_hits[lvl] = (xy_in.float() * m.unsqueeze(-1)).sum(dim=(1, 2))
        pt_in = (err <= math.log(1 / (1 - math.sqrt(q)))).all(dim=-1)
        pt_hits[lvl] = (pt_in.float() * m).sum(dim=1)
    return pt_hits, xy_hits, m.sum(dim=1)
