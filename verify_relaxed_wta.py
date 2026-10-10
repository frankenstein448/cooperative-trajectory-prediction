"""
verify_relaxed_wta.py
Unit check for RelaxedWTALoss, no data/model needed:
  1. eps=0 equals WinnerTakesAllLoss EXACTLY (torch.equal on total, reg, cls,
     and identical gradients w.r.t. trajs and conf) on random batches with a
     random partially-invalid gt_mask.
  2. eps=0.05 matches a slow per-sample/per-mode reference implementation.
Run in gpu-test-venv.
"""
import torch
import torch.nn.functional as F

from trajectory_loss import WinnerTakesAllLoss, RelaxedWTALoss, RELAXED_WTA_EPS

torch.manual_seed(0)
B, K, H = 8, 6, 30


def random_batch():
    trajs = torch.randn(B, K, H, 2) * 5
    conf = torch.softmax(torch.randn(B, K), dim=-1)
    gt_traj = torch.randn(B, H, 2) * 5
    gt_mask = torch.rand(B, H) > 0.2
    gt_mask[0] = False  # one fully-invalid sample, exercises the clamp
    return trajs, conf, gt_traj, gt_mask


def loss_and_grads(loss_fn, trajs, conf, gt_traj, gt_mask):
    t = trajs.clone().requires_grad_(True)
    c = conf.clone().requires_grad_(True)
    loss, info = loss_fn(t, c, gt_traj, gt_mask)
    loss.backward()
    return loss.detach(), info, t.grad, c.grad


N_TRIALS = 20
for trial in range(N_TRIALS):
    batch = random_batch()
    lw, iw, gtw, gcw = loss_and_grads(WinnerTakesAllLoss(), *batch)
    lr, ir, gtr, gcr = loss_and_grads(RelaxedWTALoss(eps=0.0), *batch)
    assert torch.equal(lw, lr), f"trial {trial}: total {lw.item()!r} vs {lr.item()!r}"
    assert iw["reg_loss"] == ir["reg_loss"] and iw["cls_loss"] == ir["cls_loss"], f"trial {trial}: components differ"
    assert iw["winner_idx"] == ir["winner_idx"], f"trial {trial}: winner_idx differs"
    assert torch.equal(gtw, gtr) and torch.equal(gcw, gcr), f"trial {trial}: gradients differ"
print(f"OK 1: eps=0 RelaxedWTALoss == WinnerTakesAllLoss exactly (loss, reg, cls, winner, grads) "
      f"on {N_TRIALS} random batches [B={B}, K={K}, H={H}]")


def reference(trajs, conf, gt_traj, gt_mask, eps):
    """Slow, loop-based statement of the spec."""
    winner = WinnerTakesAllLoss()(trajs, conf, gt_traj, gt_mask)[1]["winner_idx"]
    norm = max(int(gt_mask.sum()) * 2, 1)
    reg = 0.0
    for b in range(B):
        for k in range(K):
            w = (1 - eps) if k == winner[b] else eps / (K - 1)
            per = F.smooth_l1_loss(trajs[b, k], gt_traj[b], reduction="none") * gt_mask[b].unsqueeze(-1)
            reg += w * per.sum().item()
    reg /= norm
    cls = F.cross_entropy(conf, torch.tensor(winner)).item()
    return reg + cls


for trial in range(N_TRIALS):
    batch = random_batch()
    got = RelaxedWTALoss(eps=RELAXED_WTA_EPS)(*batch)[0].item()
    want = reference(*batch, eps=RELAXED_WTA_EPS)
    assert abs(got - want) < 1e-4 * max(1.0, abs(want)), f"trial {trial}: {got} vs reference {want}"
print(f"OK 2: eps={RELAXED_WTA_EPS} RelaxedWTALoss matches the loop reference on {N_TRIALS} random batches")
