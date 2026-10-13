"""
export_lane_polylines.py
Lane polylines for the env task's map tokens (runs/DESIGN_ENV.md, M1). For ONE town per call, from the
live CARLA server (started with that town as its startup map; load_world is not used, as in
dense_waypoints.py), it takes generate_waypoints(2.0), the same call that built dense_waypoints_<town>.npy, with:
x, y, yaw (deg), road_id, section_id, lane_id, s, is_junction, lane_width, lane type.

Check (hard failure): the driving points must reproduce dense_waypoints_<town>.npy exactly (same count,
same order, identical float64 coordinates).

--include-parking: also Parking lanes. For every driving waypoint, the lanes beside it are walked with
get_left_lane() / get_right_lane() (up to MAX_SIDE_STEPS each way, through any lane type) and every Parking
waypoint found is kept (same s as its driving neighbour, so ~2 m spacing), deduplicated by
(road, section, lane, s). CARLA 0.9.15 types some lanes that V2X-Sim's SUMO traffic drives on as Parking
(Town03: 98% of the moving vehicles > 2.5 m from every driving waypoint stand on one; see
check_map_alignment.log). The driving points come first, in generate_waypoints order.

Polylines: waypoints grouped by (lane_type, road_id, section_id, lane_id), ordered along the direction of
travel (sorted by s, reversed if the steps point against the waypoint yaw), and cut into consecutive pieces
of PIECE_POINTS points (about 20 m). A trailing piece shorter than 2 points is merged into the previous one.

Writes lane_polylines_<town>.npz:
  points [N, 2] float64 (CARLA x, y; points[:n_driving] == dense_waypoints), lane_type [N] str
  ("Driving" / "Parking"), n_driving, yaw_deg [N], road_id / section_id / lane_id [N], s [N],
  is_junction [N] bool, lane_width [N];
  order [N] int64 (point indices, piece after piece, each in travel order), piece_ptr [P + 1] int64
  (piece p = order[piece_ptr[p]:piece_ptr[p + 1]]), piece_lane [P, 3] (road, section, lane), piece_type [P] str.

Needs the carla package:  ~/carla-venv/bin/python export_lane_polylines.py Town03 [--include-parking]
"""
import sys
import collections

import numpy as np
import carla

PIECE_POINTS = 10
MAX_SIDE_STEPS = 4
TOWN = sys.argv[1]
INCLUDE_PARKING = "--include-parking" in sys.argv[2:]

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
cmap = client.get_world().get_map()
loaded = cmap.name.split("/")[-1]
if TOWN not in loaded:
    raise RuntimeError(f"server has {loaded!r} loaded, expected {TOWN!r} (start CARLA with the town as startup map)")

wps = list(cmap.generate_waypoints(2.0))
n_driving = len(wps)
drv = np.array([[w.transform.location.x, w.transform.location.y] for w in wps], dtype=np.float64)
ref = np.load(f"dense_waypoints_{TOWN.lower()}.npy")
if drv.shape != ref.shape or not np.array_equal(drv, ref):
    raise RuntimeError(f"{TOWN}: export does not reproduce dense_waypoints exactly ({len(drv)} vs {len(ref)} points)")
print(f"{TOWN}: {n_driving} driving waypoints, identical (count, order, float64 values) to "
      f"dense_waypoints_{TOWN.lower()}.npy")

if INCLUDE_PARKING:
    seen = set()
    for w in wps[:n_driving]:
        for side in ("get_left_lane", "get_right_lane"):
            cur = w
            for _ in range(MAX_SIDE_STEPS):
                cur = getattr(cur, side)()
                if cur is None:
                    break
                if cur.lane_type == carla.LaneType.Parking:
                    key = (cur.road_id, cur.section_id, cur.lane_id, round(cur.s, 2))
                    if key not in seen:
                        seen.add(key)
                        wps.append(cur)
    print(f"  + {len(wps) - n_driving} Parking waypoints beside driving ones")

pts = np.array([[w.transform.location.x, w.transform.location.y] for w in wps], dtype=np.float64)
assert np.array_equal(pts[:n_driving], ref)
ltype = np.array(["Driving"] * n_driving + ["Parking"] * (len(wps) - n_driving))
yaw = np.array([w.transform.rotation.yaw for w in wps])
road = np.array([w.road_id for w in wps], dtype=np.int64)
sec = np.array([w.section_id for w in wps], dtype=np.int64)
lane = np.array([w.lane_id for w in wps], dtype=np.int64)
s = np.array([w.s for w in wps])
junc = np.array([w.is_junction for w in wps], dtype=bool)
width = np.array([w.lane_width for w in wps])

groups = collections.defaultdict(list)
for i in range(len(wps)):
    groups[(ltype[i], road[i], sec[i], lane[i])].append(i)

order, piece_ptr, piece_lane, piece_type = [], [0], [], []
n_rev = n_single = n_dup_s = 0
step_len, step_vs_yaw = [], []
for key in sorted(groups):
    idx = sorted(groups[key], key=lambda i: s[i])
    n_dup_s += len(idx) - len({round(s[i], 6) for i in idx})
    if len(idx) >= 2:
        d = pts[idx[1:]] - pts[idx[:-1]]
        h = np.radians(yaw[idx[:-1]])
        if np.mean(d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)) < 0:
            idx = idx[::-1]
            n_rev += 1
        d = pts[idx[1:]] - pts[idx[:-1]]
        step_len += list(np.hypot(d[:, 0], d[:, 1]))
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) - yaw[idx[:-1]]
        step_vs_yaw += list(np.abs((ang + 180) % 360 - 180))
    else:
        n_single += 1
    cuts = list(range(0, len(idx), PIECE_POINTS))
    if len(cuts) > 1 and len(idx) - cuts[-1] < 2:
        cuts.pop()
    for a, b in zip(cuts, cuts[1:] + [len(idx)]):
        order += idx[a:b]
        piece_ptr.append(len(order))
        piece_lane.append(key[1:])
        piece_type.append(key[0])

order = np.array(order, dtype=np.int64)
assert np.array_equal(np.sort(order), np.arange(len(pts))), "every waypoint must be in exactly one piece"
step_len, step_vs_yaw = np.array(step_len), np.array(step_vs_yaw)
plen = np.diff(piece_ptr)
print(f"  lanes (type, road, section, lane): {len(groups)} ({sum(k[0] == 'Parking' for k in groups)} Parking); "
      f"reversed to travel order: {n_rev}; single-point lanes: {n_single}; duplicate s within a lane: {n_dup_s}")
print(f"  pieces: {len(plen)} ({sum(t == 'Parking' for t in piece_type)} Parking); points per piece: min {plen.min()} "
      f"p50 {np.median(plen):.0f} max {plen.max()}; in-junction points {junc.mean():.3f}")
print(f"  consecutive step length (m): p1 {np.percentile(step_len, 1):.2f} p50 {np.median(step_len):.2f} "
      f"p99 {np.percentile(step_len, 99):.2f} max {step_len.max():.2f}; step direction vs yaw: p50 "
      f"{np.median(step_vs_yaw):.2f} deg, p99 {np.percentile(step_vs_yaw, 99):.2f}, >90 deg {np.mean(step_vs_yaw > 90):.4f}")
np.savez(f"lane_polylines_{TOWN.lower()}.npz", points=pts, lane_type=ltype, n_driving=n_driving, yaw_deg=yaw,
         road_id=road, section_id=sec, lane_id=lane, s=s, is_junction=junc, lane_width=width, order=order,
         piece_ptr=np.array(piece_ptr, dtype=np.int64), piece_lane=np.array(piece_lane, dtype=np.int64),
         piece_type=np.array(piece_type))
print(f"  wrote lane_polylines_{TOWN.lower()}.npz")
