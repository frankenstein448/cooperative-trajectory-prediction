"""
verify_loss.py
Unit checks for --loss laplace_mixture / wta_soft and the ranking metrics, on random
tensors only (no PilotDataset). Runs on CPU.

  1. LaplaceMixtureLoss: finite loss and finite, non-zero gradients w.r.t. trajs,
     conf and scales on random batches (partially-invalid masks, one fully-invalid
     sample, which must contribute exactly 0 and no gradient); also end-to-end
     through TrajectoryHead(predict_scale=True) (gradient reaches scale_mlp).
  2. With one mode, LaplaceMixtureLoss == the masked single-Laplace NLL
     (torch.distributions.Laplace), per sample summed over valid steps and x/y.
  3. Slow per-sample / per-mode reference for the 6-mode mixture.
  4. WTASoftLoss: finite loss/gradients; regression term identical to WTA's; as
     tau -> 0 the soft target becomes WTA's one-hot winner and the classification
     loss approaches WTA's cross-entropy.
  5. Model: predict_scale=False is D (strict load of the D checkpoint, same parameter
     count); predict_scale=True adds exactly scale_mlp (D's state_dict loads with only
     scale_mlp.* missing), scales are > LAPLACE_SCALE_FLOOR, and the global RNG
     stream after construction is the same as without it.
  6. Ranking metrics on hand-computed cases; Laplace coverage of samples drawn from
     the predicted distribution is close to nominal.

  python verify_loss.py [--d-ckpt runs/D_seed0/checkpoint_epoch12.pt]
Run in gpu-test-venv.
"""
import argparse
import math
import torch
import torch.nn.functional as F

from trajectory_loss import (WinnerTakesAllLoss, LaplaceMixtureLoss, WTASoftLoss, build_loss,
                             LAPLACE_SCALE_FLOOR, WTA_SOFT_TAU)
from end_to_end_model import EndToEndFusion, TrajectoryHead, EMBED_DIM
from ranking_metrics import brier_min_fde, mode_diversity, laplace_coverage

parser = argparse.ArgumentParser()
parser.add_argument("--d-ckpt", default="runs/D_seed0/checkpoint_epoch12.pt")
args = parser.parse_args()

torch.manual_seed(0)
B, K, H = 8, 6, 30


def random_batch(k=K):
    trajs = torch.randn(B, k, H, 2) * 5
    conf = torch.randn(B, k)
    scales = F.softplus(torch.randn(B, k, H, 2)) + LAPLACE_SCALE_FLOOR
    gt_traj = torch.randn(B, H, 2) * 5
    gt_mask = torch.rand(B, H) > 0.2
    gt_mask[0] = False  # one fully-invalid sample
    return trajs, conf, scales, gt_traj, gt_mask


def finite_grads(loss, *tensors):
    grads = torch.autograd.grad(loss, tensors, allow_unused=False)
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient"
    return grads


# ---------- 1. laplace_mixture: finite loss and gradients ----------
lap = build_loss("laplace_mixture")
assert isinstance(lap, LaplaceMixtureLoss)
for trial in range(20):
    trajs, conf, scales, gt, mask = random_batch()
    if trial == 1:
        scales = torch.full_like(scales, LAPLACE_SCALE_FLOOR)  # smallest scale the model can give
    t, c, s = (x.clone().requires_grad_(True) for x in (trajs, conf, scales))
    loss, info = lap(t, c, gt, mask, s)
    assert torch.isfinite(loss), f"trial {trial}: loss {loss}"
    gt_, gc, gs = finite_grads(loss, t, c, s)
    # conf: -logsumexp(log_softmax(c)) == 0 exactly in theory; float residue only
    assert gt_[0].abs().sum() == 0 and gs[0].abs().sum() == 0 and gc[0].abs().max() < 1e-6, \
        f"fully-invalid sample got gradient (conf {gc[0].abs().max():.2e})"
    assert gt_.abs().sum() > 0 and gc.abs().sum() > 0 and gs.abs().sum() > 0
    # invalid steps get no location / scale gradient
    inv = ~mask
    assert gt_.permute(0, 2, 1, 3)[inv].abs().sum() == 0 and gs.permute(0, 2, 1, 3)[inv].abs().sum() == 0
# the fully-invalid sample contributes exactly 0 to the batch sum
trajs, conf, scales, gt, mask = random_batch()
full = lap(trajs, conf, gt, mask, scales)[0] * B
rest = lap(trajs[1:], conf[1:], gt[1:], mask[1:], scales[1:])[0] * (B - 1)
assert torch.allclose(full, rest, atol=1e-4), (full, rest)

head = TrajectoryHead(predict_scale=True)
x = torch.randn(B, EMBED_DIM)
h_trajs, h_conf, h_scales = head(x)
assert h_scales.shape == h_trajs.shape and (h_scales > LAPLACE_SCALE_FLOOR).all()
loss, _ = lap(h_trajs, h_conf, gt, mask, h_scales)
loss.backward()
for name, p in head.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
print(f"1. laplace_mixture: finite loss and gradients on 20 random batches (incl. all scales at the "
      f"{LAPLACE_SCALE_FLOOR} m floor); fully-invalid sample / invalid steps get 0 loss and 0 gradient; "
      f"gradient reaches traj_mlp, conf_mlp, scale_mlp")

# ---------- 2. one mode == single Laplace NLL ----------
for _ in range(5):
    trajs, conf, scales, gt, mask = random_batch(k=1)
    loss, _ = lap(trajs, conf, gt, mask, scales)
    lp = torch.distributions.Laplace(trajs[:, 0], scales[:, 0]).log_prob(gt)  # [B, H, 2]
    ref = -(lp * mask.unsqueeze(-1)).sum(dim=(1, 2)).mean()
    assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-4), (loss, ref)
print("2. one mode: laplace_mixture == masked single-Laplace NLL (torch.distributions.Laplace), any mode score")

# ---------- 3. slow reference, 6 modes ----------
trajs, conf, scales, gt, mask = random_batch()
per_sample = []
for b in range(B):
    terms = []
    for k in range(K):
        lp = 0.0
        for t_ in range(H):
            if mask[b, t_]:
                for d in range(2):
                    bb = scales[b, k, t_, d].item()
                    lp += -math.log(2 * bb) - abs(gt[b, t_, d].item() - trajs[b, k, t_, d].item()) / bb
        terms.append(F.log_softmax(conf[b], dim=-1)[k].item() + lp)
    m = max(terms)
    per_sample.append(-(m + math.log(sum(math.exp(v - m) for v in terms))))
ref = sum(per_sample) / B
got = lap(trajs, conf, gt, mask, scales)[0].item()
assert abs(got - ref) < 1e-3 * max(1.0, abs(ref)), (got, ref)
print(f"3. 6 modes: matches the per-sample/per-mode reference ({got:.4f} vs {ref:.4f})")

# ---------- 4. wta_soft ----------
soft = build_loss("wta_soft")
assert isinstance(soft, WTASoftLoss) and soft.tau == WTA_SOFT_TAU == 1.0
wta = WinnerTakesAllLoss()
for _ in range(10):
    trajs, conf, _, gt, mask = random_batch()
    t, c = trajs.clone().requires_grad_(True), conf.clone().requires_grad_(True)
    loss, info = soft(t, c, gt, mask)
    assert torch.isfinite(loss)
    finite_grads(loss, t, c)
    w_loss, w_info = wta(trajs, conf, gt, mask)
    assert info["reg_loss"] == w_info["reg_loss"] and info["winner_idx"] == w_info["winner_idx"]
    # regression gradient w.r.t. trajs identical to WTA's (cls term has no trajs gradient: ADE is detached)
    t1, t2 = trajs.clone().requires_grad_(True), trajs.clone().requires_grad_(True)
    g_soft = torch.autograd.grad(soft(t1, conf, gt, mask)[0], t1)[0]
    g_wta = torch.autograd.grad(wta(t2, conf, gt, mask)[0], t2)[0]
    assert torch.equal(g_soft, g_wta)
# A sample with no valid step has ADE 0 for every mode: WTA's argmin picks mode 0, the soft
# target is uniform. The tau -> 0 limit is therefore checked on samples with valid steps.
trajs, conf, _, gt, mask = random_batch()
s_info = soft(trajs, conf, gt, mask)[1]
assert torch.allclose(F.softmax(-s_info["ade_per_mode"][0] / soft.tau, dim=-1), torch.full((K,), 1 / K))
trajs, conf, gt, mask = trajs[1:], conf[1:], gt[1:], mask[1:]
w_cls = wta(trajs, conf, gt, mask)[1]["cls_loss"]
winner = torch.tensor(wta(trajs, conf, gt, mask)[1]["winner_idx"])
for tau in (1.0, 0.1, 0.01, 1e-3, 1e-4):
    s_info = WTASoftLoss(tau=tau)(trajs, conf, gt, mask)[1]
    target = F.softmax(-s_info["ade_per_mode"] / tau, dim=-1)
    gap = abs(s_info["cls_loss"] - w_cls)
    print(f"   tau={tau:<6g}: max |soft target - one-hot winner| = "
          f"{(target - F.one_hot(winner, K)).abs().max():.2e}, |cls - WTA cls| = {gap:.2e}")
assert torch.allclose(target, F.one_hot(winner, K).float(), atol=1e-6) and gap < 1e-5
print("4. wta_soft: finite loss/gradients; regression term and its trajs gradient identical to WTA; "
      "tau -> 0 gives WTA's one-hot target and cross-entropy (samples with valid steps; a fully-invalid "
      "sample gets a uniform target where WTA picks mode 0)")

# ---------- 5. model ----------
n_params = lambda m: sum(p.numel() for p in m.parameters())
torch.manual_seed(1)
d = EndToEndFusion(device="cpu")
after_d = torch.rand(5)
torch.manual_seed(1)
lm = EndToEndFusion(device="cpu", predict_scale=True)
after_l = torch.rand(5)
assert torch.equal(after_d, after_l), "scale head shifted the global RNG stream"
n_scale = n_params(lm.traj_head.scale_mlp)
assert n_scale == EMBED_DIM * K * H * 2 + K * H * 2
assert n_params(lm) == n_params(d) + n_scale
sd_l = lm.state_dict()
for k_, v in d.state_dict().items():
    assert torch.equal(v, sd_l[k_]), f"{k_} init differs with predict_scale=True"
assert set(sd_l) - set(d.state_dict()) == {"traj_head.scale_mlp.weight", "traj_head.scale_mlp.bias"}
ckpt = torch.load(args.d_ckpt, map_location="cpu")["model_state_dict"]
d.load_state_dict(ckpt, strict=True)
res = lm.load_state_dict(ckpt, strict=False)
assert set(res.missing_keys) == {"traj_head.scale_mlp.weight", "traj_head.scale_mlp.bias"} and not res.unexpected_keys
print(f"5. model: predict_scale=False strict-loads {args.d_ckpt} ({n_params(d):,} parameters); "
      f"predict_scale=True = D + scale_mlp ({n_scale:,}), identical init of every D parameter and the same RNG "
      f"stream afterwards; D's state_dict loads with only scale_mlp missing")

# ---------- 6. ranking metrics ----------
tr = torch.zeros(1, 3, H, 2)
tr[0, 1, -1] = torch.tensor([3.0, 4.0])   # 5 m from mode 0
tr[0, 2, -1] = torch.tensor([6.0, 8.0])   # 10 m from mode 0, 5 m from mode 1
assert torch.allclose(mode_diversity(tr), torch.tensor([(5 + 10 + 5) / 3]))
fde = torch.tensor([[2.0, 1.0, 3.0]])
sc = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
assert torch.allclose(brier_min_fde(fde, sc), torch.tensor([1.0 + 0.25]))
torch.manual_seed(2)
mu = torch.randn(4, K, H, 2)
b = F.softplus(torch.randn(4, K, H, 2)) + LAPLACE_SCALE_FLOOR
top = torch.tensor([0, 3, 5, 2])
n_draw, pts, xys, n_pts = 2000, {50: 0.0, 90: 0.0}, {50: 0.0, 90: 0.0}, 0.0
msk = torch.rand(4, H) > 0.3
for _ in range(n_draw // 4):
    g = torch.distributions.Laplace(mu[torch.arange(4), top], b[torch.arange(4), top]).sample()
    p_hits, x_hits, npt = laplace_coverage(mu, b, g, msk, top)
    for l in (50, 90):
        pts[l] += p_hits[l].sum().item()
        xys[l] += x_hits[l].sum().item()
    n_pts += npt.sum().item()
for l in (50, 90):
    pc, xc = pts[l] / n_pts, xys[l] / (2 * n_pts)
    assert abs(pc - l / 100) < 0.01 and abs(xc - l / 100) < 0.01, (l, pc, xc)
    print(f"   coverage of {int(n_pts)} sampled GT points, nominal {l}%: pt {pc:.4f}  xy {xc:.4f}")
print("6. ranking metrics: diversity and Brier-minFDE match hand-computed values; Laplace coverage is nominal "
      "on samples from the predicted distribution")
print("\nALL LOSS CHECKS PASSED")
