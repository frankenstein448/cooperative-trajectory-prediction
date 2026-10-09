import pickle
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader

TOWN = "Town04"
A0, AX, AY = 260.50, 37.75, -41.25   # earlier coarse-search candidate A
CAP = 10.0


def rot(deg):
    th = np.radians(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def pose_xy(loader, agent_id, scene, frame):
    rec = loader.find_sample_data(agent_id, scene, frame)
    tr = loader.ego_pose[rec["ego_pose_token"]]["translation"]
    return np.array([tr[0], tr[1]])


def lane_points(topo, step=1.0):
    pts = []
    for e in topo:
        if e["relation"] != 0:
            continue
        a, b = np.array(e["start_pos"], float), np.array(e["end_pos"], float)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts)


dataset = PilotDataset()
loader = V2XMetadataLoader()
topo = pickle.load(open(f"topology/{TOWN}_topology.pkl", "rb"))
tree = cKDTree(lane_points(topo))

frames = defaultdict(set)
for agent_id, scene, frame, tn in dataset.samples:
    if tn == TOWN:
        frames[(agent_id, scene)].add(frame)

pts_l, aid_l, moving_flag = [], [], []
for (agent_id, scene), fs in frames.items():
    p = []
    for f in sorted(fs):
        try:
            p.append(pose_xy(loader, agent_id, scene, f))
        except Exception:
            pass
    p = np.array(p)
    if len(p) < 1:
        continue
    length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum() if len(p) > 1 else 0.0
    moving = length >= 10.0
    if not moving:
        p = p.mean(0, keepdims=True)
    k = len(moving_flag)
    pts_l.append(p)
    aid_l.append(np.full(len(p), k))
    moving_flag.append(moving)

pts = np.vstack(pts_l)
aid = np.concatenate(aid_l)
moving_flag = np.array(moving_flag)
n_agents = len(moving_flag)
counts = np.bincount(aid, minlength=n_agents)
pt_moving = moving_flag[aid]
print(f"{TOWN}: {int(moving_flag.sum())} moving, {int((~moving_flag).sum())} stationary agents, {len(pts)} points")

c_raw = pts.mean(0)
c_t = c_raw @ rot(A0).T + np.array([AX, AY])


def to_world(dth, dx, dy):
    return (pts - c_raw) @ rot(A0 + dth).T + c_t + np.array([dx, dy])


def per_agent_cost(dth, dx, dy):
    d, _ = tree.query(to_world(dth, dx, dy))
    return np.bincount(aid, weights=np.minimum(d, CAP), minlength=n_agents) / counts


def constants(dth, dx, dy):
    th = A0 + dth
    t = c_t + np.array([dx, dy]) - rot(th) @ c_raw
    return th, t[0], t[1]


def report(name, dth, dx, dy):
    d, _ = tree.query(to_world(dth, dx, dy))
    th, tx, ty = constants(dth, dx, dy)
    mv, st = d[pt_moving], d[~pt_moving]
    print(f"  {name:8s} theta={th:7.2f} tx={tx:8.2f} ty={ty:8.2f} | "
          f"moving median {np.median(mv):.2f} m, <2m {np.mean(mv < 2) * 100:.0f}% | "
          f"stationary median {np.median(st):.2f} m, <2m {np.mean(st < 2) * 100:.0f}%")


# coarse grid, per-agent costs stored so the bootstrap only re-weights
dths = np.arange(-8, 8.01, 0.5)
dxs = np.arange(-8, 8.01, 1.0)
dys = np.arange(-8, 8.01, 1.0)
T = np.zeros((len(dths), len(dxs), len(dys), n_agents))
for i, a in enumerate(dths):
    for j, x in enumerate(dxs):
        for k, y in enumerate(dys):
            T[i, j, k] = per_agent_cost(a, x, y)
flat = T.reshape(-1, n_agents)
best = flat.mean(1).argmin()
bi, bj, bk = np.unravel_index(best, T.shape[:3])
dth0, dx0, dy0 = dths[bi], dxs[bj], dys[bk]

# fine search around the coarse best
best_c, best_p = 1e9, None
for a in np.arange(dth0 - 0.5, dth0 + 0.51, 0.1):
    for x in np.arange(dx0 - 1, dx0 + 1.01, 0.25):
        for y in np.arange(dy0 - 1, dy0 + 1.01, 0.25):
            c = per_agent_cost(a, x, y).mean()
            if c < best_c:
                best_c, best_p = c, (a, x, y)

print("\nResults (all agents joint fit, so no held-out data here):")
report("A", 0.0, 0.0, 0.0)
report("fit", *best_p)

th, tx, ty = constants(*best_p)
print(f'\n  suggested: "theta": {th:.2f}, "tx": {tx:.2f}, "ty": {ty:.2f}')

# bootstrap over agents
rng = np.random.default_rng(0)
thetas = []
for _ in range(40):
    w = np.bincount(rng.integers(0, n_agents, n_agents), minlength=n_agents).astype(float)
    idx = (flat @ w).argmin()
    i, _, _ = np.unravel_index(idx, T.shape[:3])
    thetas.append(A0 + dths[i])
thetas = np.array(thetas)
print(f"\nBootstrap (40 resamples of agents): theta median {np.median(thetas):.1f}, "
      f"std {thetas.std():.1f}, range {thetas.min():.1f} to {thetas.max():.1f}")

# split-half validation: fit on half the agents, score on the other half
rng = np.random.default_rng(1)
i0, j0, k0 = np.abs(dths).argmin(), np.abs(dxs).argmin(), np.abs(dys).argmin()
res = []
for _ in range(20):
    perm = rng.permutation(n_agents)
    tr, te = perm[: n_agents // 2], perm[n_agents // 2:]
    idx = flat[:, tr].mean(1).argmin()
    res.append((flat[idx, te].mean(), T[i0, j0, k0, te].mean()))
res = np.array(res)
print(f"\nSplit-half: held-out cost fit {res[:, 0].mean():.2f} m vs A {res[:, 1].mean():.2f} m; "
      f"fit better in {(res[:, 0] < res[:, 1]).mean() * 100:.0f}% of 20 splits")