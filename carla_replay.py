"""
carla_replay.py
CARLA 0.9.15 replay of recorded V2X-Sim 2.0 data (visualisation only; nothing is simulated): the targets of
export_carla_replay.py's per-frame .npz are placed with set_transform (physics off, no traffic manager, synchronous
mode at 0.2 s) and an RGB camera behind / above the ego renders the scene. Predictions are drawn later by
overlay_carla_replay.py with the saved camera matrices.

  smoke   start CARLA with the town as startup map (tmux session carla_replay, -RenderOffScreen -opengl|-vulkan (--rhi)
          -quality-level=Low|Epic (--quality); client.load_world() segfaults on this machine, as dense_waypoints.py
          found, so the town is set in DefaultEngine.ini for the start and restored from DefaultEngine.ini.orig_replay),
          spawn one frame, save frameNNN.png + camera_frameNNN.json, report memory, rendered paint colours and lane
          fit, shut CARLA down.
  render  the same for every exported frame (frameNNN.png + camera_frameNNN.json each), RAM checked every 10 frames;
          lane fit over all frames.

The map's built-in vehicles (CityObjectLabel Car / Truck / Motorcycle / Bicycle) are disabled.
V2X-Sim -> CARLA: (x, y, yaw) -> (x, -y, -yaw) (the mirror, all towns). z: world.ground_projection searched downward
from 3 m above the nearest map waypoint, minus the blueprint's bounding-box bottom (bb.location.z - bb.extent.z), so
the box rests on the ground. Blueprints keep their own size (CARLA cannot scale actors). Cars: CAR_BP recoloured by
visibility class (paint(): sRGB -> linear, so the render matches the legend); two-wheelers: BIKE_BP (not
recolourable; their class shows in the overlay only); ego: CAR_BP black.

Run in carla-venv from the data checkout (nothing else may run: refuses while a run / train / eval process or another
CARLA server is up):
  python ~/hd-maps-dev/carla_replay.py smoke --frame 35 [--quality Low --out <dir>]
  python ~/hd-maps-dev/carla_replay.py render
"""
import os
import re
import sys
import json
import math
import time
import glob
import queue
import argparse
import subprocess

import numpy as np
import carla

CARLA_DIR = os.path.expanduser("~/carla_sim")
ENGINE_INI = os.path.join(CARLA_DIR, "CarlaUE4/Config/DefaultEngine.ini")
ENGINE_BAK = ENGINE_INI + ".orig_replay"   # the original, made on the first start and kept
MAP_LINES = re.compile(r"^((?:EditorStartupMap|GameDefaultMap|ServerDefaultMap)=/Game/Carla/Maps/)\w+\.\w+$", re.M)
TMUX = "carla_replay"
PORT = 2000
DT = 0.2
CAR_BP, BIKE_BP = "vehicle.lincoln.mkz_2020", "vehicle.diamondback.century"
CLASS_HEX = ("#0072B2", "#E69F00", "#009E73", "#8C8C8C")   # render_env_bev.CLASS_COLORS (Okabe-Ito)


def paint(hex_color):
    """Legend colour (sRGB hex) -> CARLA 'color' attribute: the sRGB -> linear transfer applied beforehand, as the
    paint material treats the attribute as linear and would otherwise render it lighter."""
    out = []
    for i in (1, 3, 5):
        c = int(hex_color[i:i + 2], 16) / 255.0
        lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        out.append(str(int(round(255 * lin))))
    return ",".join(out)


CLASS_RGB = tuple(paint(h) for h in CLASS_HEX)
EGO_RGB = "0,0,0"
IMG_W, IMG_H, FOV = 1920, 1080, 90.0
CAM_BACK, CAM_UP, CAM_PITCH = 25.0, 30.0, -50.0
WARMUP_TICKS = 10          # let textures / LODs stream in before the saved image
FRAME_TICKS = 4            # per later frame: temporal anti-aliasing settles after the actors are moved
HIDE_LABELS = ("Car", "Truck", "Motorcycle", "Bicycle")
RAM_ABORT = 0.85
BUSY = re.compile(r"(run_\w+\.sh|train\.py|eval_\w+\.py)")


def busy():
    """Run / train / eval processes (by args) and CARLA servers (by process name), excluding this process and its
    ancestors (a calling shell's command line may contain these names)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="], capture_output=True, text=True).stdout
    procs = {int(l.split()[0]): (int(l.split()[1]), l.split()[2], l.strip()) for l in out.splitlines()}
    mine, p = set(), os.getpid()
    while p in procs and p not in mine:
        mine.add(p)
        p = procs[p][0]
    return [line for pid, (_, comm, line) in procs.items()
            if pid not in mine and (BUSY.search(line) or comm.startswith("CarlaUE4"))]


def ram_used():
    m = {}
    with open("/proc/meminfo") as f:
        for l in f:
            k, v = l.split(":")
            m[k] = int(v.split()[0])
    return 1.0 - m["MemAvailable"] / m["MemTotal"]


def mem_report(tag):
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"--- memory [{tag}]: RAM used {100 * ram_used():.1f}% | GPU {gpu}\n{free}")


def read_text(path):
    with open(path) as f:
        return f.read()


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def engine_ini_original():
    """-> the original DefaultEngine.ini text. The first start saves it to ENGINE_BAK. Later starts compare: a live
    file that differs from the backup only in the default-map lines is a stale edit left by a crash and is restored;
    any other difference refuses (changed by hand: check, then delete ENGINE_BAK)."""
    live = read_text(ENGINE_INI)
    if not os.path.exists(ENGINE_BAK):
        write_text(ENGINE_BAK, live)
        print(f"DefaultEngine.ini backed up to {ENGINE_BAK}")
        return live
    orig = read_text(ENGINE_BAK)
    if live != orig:
        if MAP_LINES.sub("", live) != MAP_LINES.sub("", orig):
            sys.exit(f"{ENGINE_INI} differs from {ENGINE_BAK} outside the default-map lines; refusing (check by hand)")
        write_text(ENGINE_INI, orig)
        print(f"stale DefaultEngine.ini edit (default map {re.search(r'GameDefaultMap=(.*)', live).group(1)}) found "
              f"and restored from {ENGINE_BAK}")
    return orig


def start_carla(rhi, quality, town):
    """Start CARLA with `town` as its startup map: DefaultEngine.ini's default-map lines are set to the town for the
    start and restored from the backup right after the world is up (load_world() and a map argument do not work on
    this machine)."""
    orig = engine_ini_original()
    ini = MAP_LINES.sub(rf"\g<1>{town}.{town}", orig)
    assert len(re.findall(rf"Maps/{town}\.{town}$", ini, flags=re.M)) == 3, "DefaultEngine.ini: 3 default-map lines"
    write_text(ENGINE_INI, ini)
    try:
        return _start_carla(rhi, quality)
    finally:
        write_text(ENGINE_INI, orig)
        assert read_text(ENGINE_INI) == read_text(ENGINE_BAK), "DefaultEngine.ini restore failed"
        print(f"DefaultEngine.ini restored, identical to {ENGINE_BAK} "
              f"(default map {re.search(r'GameDefaultMap=(.*)', orig).group(1)})")


def _start_carla(rhi, quality):
    subprocess.run(["tmux", "new-session", "-d", "-s", TMUX,
                    f"cd {CARLA_DIR} && ./CarlaUE4.sh -RenderOffScreen -{rhi} -quality-level={quality} "
                    f"-carla-rpc-port={PORT} 2>&1 | tee /tmp/carla_replay_server.log"], check=True)
    t0 = time.time()
    while True:
        # a Client made before the port listens can stay disconnected: wait for the port, a fresh Client per try
        listening = "LISTEN" in subprocess.run(["ss", "-ltn", f"sport = :{PORT}"], capture_output=True,
                                               text=True).stdout
        try:
            if listening:
                client = carla.Client("localhost", PORT)
                client.set_timeout(10.0)
                print(f"CARLA server {client.get_server_version()} up after {time.time() - t0:.0f} s")
                break
        except RuntimeError:
            pass
        if time.time() - t0 > 300:   # a cold first start compiles shaders
            stop_carla()
            sys.exit("CARLA did not start within 300 s")
        time.sleep(3)
    client.set_timeout(180.0)
    return client


def stop_carla():
    subprocess.run(["tmux", "kill-session", "-t", TMUX], capture_output=True)
    subprocess.run(["pkill", "CarlaUE4"], capture_output=True)
    for _ in range(30):
        if not subprocess.run(["pgrep", "CarlaUE4"], capture_output=True).stdout:
            break
        time.sleep(1)
    left = subprocess.run(["pgrep", "-a", "CarlaUE4"], capture_output=True, text=True).stdout.strip()
    port = subprocess.run(["ss", "-ltn", f"sport = :{PORT}"], capture_output=True,
                          text=True).stdout.strip().splitlines()
    print(f"CARLA shut down: processes left {left or 'none'}; port {PORT} "
          f"{'still listening' if len(port) > 1 else 'free'}")


def load_town(client, town):
    world = client.get_world()   # the startup map (start_carla)
    name = world.get_map().name
    assert name.split("/")[-1] == town, f"loaded map {name}, expected {town}"
    print(f"map {name}")
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds, s.substepping = True, DT, False   # physics is off: no substeps
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    hide_map_vehicles(world)
    world.tick()
    return world


def hide_map_vehicles(world):
    """Disable the map's built-in (parked) vehicles, so that only the replayed V2X-Sim vehicles appear."""
    n = {}
    hide = set()
    for name in HIDE_LABELS:
        ids = {o.id for o in world.get_environment_objects(getattr(carla.CityObjectLabel, name))}
        n[name] = len(ids)
        hide |= ids
    world.enable_environment_objects(hide, False)
    print("map vehicles hidden: " + ", ".join(f"{k} {v}" for k, v in n.items()))


def to_carla(x, y, yaw):
    """V2X-Sim world (x, y, yaw rad) -> CARLA (x, y, yaw deg): the mirror."""
    return x, -y, -math.degrees(yaw)


def ground_z(world, cmap, x, y):
    """CARLA x, y -> (ground z, waypoint z): ray downward from 3 m above the nearest waypoint (any lane type)."""
    wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
    wz = wp.transform.location.z
    hit = world.ground_projection(carla.Location(x, y, wz + 3.0), 10.0)
    return (hit.location.z if hit is not None else wz), wz


def load_frame(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}, json.loads(str(z["meta"]))


class Scene:
    """Spawned actors (-1: the ego); smoke keys them by tid, render by (two_wheeler, cls, index) for reuse."""

    def __init__(self, world):
        self.world, self.cmap, self.lib = world, world.get_map(), world.get_blueprint_library()
        for bp_id in (CAR_BP, BIKE_BP):
            assert self.lib.find(bp_id) is not None, bp_id
        assert self.lib.find(CAR_BP).has_attribute("color"), f"{CAR_BP} is not recolourable"
        self.actors, self.n_spawned, self.parking = {}, 0, {}

    def spawn(self, key, two_wheeler, rgb):
        bp = self.lib.find(BIKE_BP if two_wheeler else CAR_BP)
        if not two_wheeler:
            bp.set_attribute("color", rgb)
        # park far above the map until the first set_transform (no collision with other actors)
        self.n_spawned += 1
        self.parking[key] = carla.Transform(carla.Location(0, 0, 500 + 10 * self.n_spawned))
        a = self.world.spawn_actor(bp, self.parking[key])
        a.set_simulate_physics(False)
        bb = a.bounding_box
        self.actors[key] = (a, bb.location.z - bb.extent.z)
        return a

    def place(self, key, x, y, yaw):
        """V2X-Sim world pose -> actor transform; returns (ground z, waypoint z, actor z)."""
        a, bottom = self.actors[key]
        cx, cy, cyaw = to_carla(x, y, yaw)
        gz, wz = ground_z(self.world, self.cmap, cx, cy)
        a.set_transform(carla.Transform(carla.Location(cx, cy, gz - bottom), carla.Rotation(yaw=cyaw)))
        return gz, wz, gz - bottom

    def park(self, key):
        self.actors[key][0].set_transform(self.parking[key])

    def destroy(self):
        for a, _ in self.actors.values():
            a.destroy()
        self.actors = {}


def lane_fit(cmap, fr):
    """Per target: distance (m) from its CARLA position to the nearest driving-lane centre point, heading difference
    (deg, mod 180) to that lane, and whether the position lies on a driving lane."""
    out = []
    for (x, y), yaw in zip(fr["xy_world"], fr["yaw_world"]):
        cx, cy, cyaw = to_carla(x, y, yaw)
        loc = carla.Location(cx, cy, 0.0)
        wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        on = cmap.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving) is not None
        d = math.hypot(wp.transform.location.x - cx, wp.transform.location.y - cy)
        dyaw = abs((wp.transform.rotation.yaw - cyaw + 90) % 180 - 90)
        out.append((d, dyaw, on, str(wp.lane_type)))
    return out


def print_lane_fit(fit, fr):
    d = np.array([f[0] for f in fit])
    dyaw = np.array([f[1] for f in fit])
    on = np.array([f[2] for f in fit])
    mov = fr["moving"]
    for name, m in (("moving", mov), ("stationary", ~mov)):
        if not m.any():
            print(f"  {name:10s}: none")
            continue
        print(f"  {name:10s} n={m.sum():2d}: on a driving lane {on[m].sum()}/{m.sum()}; distance to the nearest "
              f"driving-lane centre median {np.median(d[m]):.2f} m, max {d[m].max():.2f} m; heading diff median "
              f"{np.median(dyaw[m]):.1f} deg, max {dyaw[m].max():.1f} deg")
    for j in np.argsort(-d)[:6]:
        print(f"    tid {fr['tid'][j]}: {'moving' if mov[j] else 'stationary'}, "
              f"{'two-wheeler' if fr['two_wheeler'][j] else 'car'}, cls {fr['cls'][j]}, d {d[j]:.2f} m, "
              f"heading diff {dyaw[j]:.1f} deg, on driving lane {on[j]}")


def spawn_frame(scene, fr):
    """Spawn every target and the ego of one frame; returns {key: (ground z, waypoint z, actor z)}."""
    z = {}
    for j, t in enumerate(fr["tid"]):
        scene.spawn(int(t), bool(fr["two_wheeler"][j]), CLASS_RGB[int(fr["cls"][j])])
        z[int(t)] = scene.place(int(t), *fr["xy_world"][j], fr["yaw_world"][j])
    ego = scene.spawn(-1, False, EGO_RGB)
    z[-1] = scene.place(-1, *fr["ego_pose"])
    return ego, z


def attach_camera(world, ego):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMG_W))
    bp.set_attribute("image_size_y", str(IMG_H))
    bp.set_attribute("fov", str(FOV))
    bp.set_attribute("motion_blur_intensity", "0.0")   # actors are teleported between frames
    cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=-CAM_BACK, z=CAM_UP), carla.Rotation(pitch=CAM_PITCH)),
                            attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
    q = queue.Queue()
    cam.listen(q.put)
    return cam, q


def capture(world, cam, q, ticks):
    """Tick `ticks` times; returns (BGRA image array of the last tick, camera->world 4x4 of that tick)."""
    for _ in range(ticks):
        frame = world.tick()
    while True:
        img = q.get(timeout=30.0)
        if img.frame == frame:
            break
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return arr.copy(), np.array(img.transform.get_matrix())


def intrinsics():
    f = IMG_W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    return [[f, 0.0, IMG_W / 2.0], [0.0, f, IMG_H / 2.0], [0.0, 0.0, 1.0]]


def project(cam2world, pts):
    """CARLA world points [N, 3] -> (pixel u, v [N, 2], depth along the camera axis [N])."""
    K = np.array(intrinsics())
    p = (np.linalg.inv(cam2world) @ np.c_[pts, np.ones(len(pts))].T)[:3]   # UE camera: x fwd, y right, z up
    return np.stack([K[0, 0] * p[1] / p[0] + K[0, 2], -K[1, 1] * p[2] / p[0] + K[1, 2]], axis=-1), p[0]


def paint_check(arr, cam2world, fr, z):
    """Rendered colour of each class's cars vs its legend colour: median of a 7x7 patch on the bonnet (1.6 m ahead of
    the box centre, 1.0 m above ground; the roof shows the interior from above), median over the cars in view."""
    cars = [j for j in range(len(fr["tid"])) if not fr["two_wheeler"][j]]
    pts = np.array([[fr["xy_world"][j, 0] + 1.6 * math.cos(fr["yaw_world"][j]),
                     -(fr["xy_world"][j, 1] + 1.6 * math.sin(fr["yaw_world"][j])),
                     z[int(fr["tid"][j])][0] + 1.0] for j in cars])
    uv, depth = project(cam2world, pts)
    for k, hex_color in enumerate(CLASS_HEX):
        rgb = [np.median(arr[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4, 2::-1].reshape(-1, 3), axis=0)
               for j, (u, v), d in zip(cars, uv, depth)
               if fr["cls"][j] == k and d > 0 and 3 <= u < IMG_W - 4 and 3 <= v < IMG_H - 4]
        got = "#%02X%02X%02X" % tuple(int(c) for c in np.median(rgb, axis=0)) if rgb else "none in view"
        print(f"  class {k} legend {hex_color}, paint attribute {CLASS_RGB[k]}: rendered {got} (n={len(rgb)} cars)")


def save_frame(arr, cam2world, fr, meta, out_dir, z, quality):
    from PIL import Image
    f = meta["frame"]
    png = os.path.join(out_dir, f"frame{f:03d}.png")
    Image.fromarray(arr[:, :, [2, 1, 0]]).save(png)
    cam = {"frame": f, "town": meta["town"], "scene": meta["scene"], "agent": meta["agent"], "width": IMG_W,
           "height": IMG_H, "quality": quality, "fov_deg": FOV, "K": intrinsics(), "cam2world": cam2world.tolist(),
           "world": "CARLA (UE, left-handed: x fwd, y right, z up, m); V2X-Sim (x, y) -> CARLA (x, -y)",
           "camera_axes": "UE camera: x forward, y right, z up; image u = fx * y/x + cx, v = -fy * z/x + cy",
           "mount": {"back_m": CAM_BACK, "up_m": CAM_UP, "pitch_deg": CAM_PITCH, "attach": "ego, rigid"},
           "ground_z": {str(k): v[0] for k, v in z.items()}}
    with open(os.path.join(out_dir, f"camera_frame{f:03d}.json"), "w") as fh:
        json.dump(cam, fh, indent=1)
    print(f"wrote {png} and camera_frame{f:03d}.json")


def cmd_smoke(args):
    path = os.path.join(args.data, f"frame_{args.frame:03d}.npz")
    fr, meta = load_frame(path)
    os.makedirs(args.out, exist_ok=True)
    mem_report("before CARLA")
    try:
        client = start_carla(args.rhi, args.quality, meta["town"])
        world = load_town(client, meta["town"])
        mem_report("Town loaded")
        scene = Scene(world)
        ego, z = spawn_frame(scene, fr)
        cam, q = attach_camera(world, ego)
        arr, cam2world = capture(world, cam, q, WARMUP_TICKS)
        mem_report("after capture")
        save_frame(arr, cam2world, fr, meta, args.out, z, args.quality)
        zz = np.array(list(z.values()))
        print(f"spawned {len(fr['tid'])} targets ({int(fr['two_wheeler'].sum())} two-wheelers) + ego; "
              f"ground z - waypoint z: median {np.median(zz[:, 0] - zz[:, 1]):.2f} m, max |.| "
              f"{np.abs(zz[:, 0] - zz[:, 1]).max():.2f} m; ground z range {zz[:, 0].min():.2f}..{zz[:, 0].max():.2f}")
        bottom = {a.type_id: b for a, b in scene.actors.values()}
        print(f"bounding-box bottom offsets: car {bottom[CAR_BP]:.3f} m, "
              + (f"bike {bottom[BIKE_BP]:.3f} m" if BIKE_BP in bottom else "no bike"))
        print(f"paint, quality {args.quality}:")
        paint_check(arr, cam2world, fr, z)
        print(f"lane fit, frame {meta['frame']}:")
        print_lane_fit(lane_fit(world.get_map(), fr), fr)
        cam.stop()
        cam.destroy()
        scene.destroy()
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        mem_report("end")
        world.apply_settings(s)
    finally:
        stop_carla()


def cmd_render(args):
    """All exported frames: actors are reused per (two-wheeler, class) and moved with set_transform (spares parked
    above the map); RAM is checked every 10 frames (abort above RAM_ABORT)."""
    paths = sorted(glob.glob(os.path.join(args.data, "frame_*.npz")))
    frames = [load_frame(p) for p in paths]
    town = frames[0][1]["town"]
    assert all(m["town"] == town and m["scene"] == frames[0][1]["scene"] for _, m in frames)
    os.makedirs(args.out, exist_ok=True)
    mem_report("before CARLA")
    fits = []
    try:
        client = start_carla(args.rhi, args.quality, town)
        world = load_town(client, town)
        mem_report("Town loaded")
        scene = Scene(world)
        ego = scene.spawn(-1, False, EGO_RGB)
        cam, q = attach_camera(world, ego)
        cmap = world.get_map()
        for n, (fr, meta) in enumerate(frames):
            if n % 10 == 0:
                r = ram_used()
                print(f"frame {meta['frame']} ({n + 1}/{len(frames)}): RAM used {100 * r:.1f}%, "
                      f"{len(scene.actors)} actors", flush=True)
                if r > RAM_ABORT:
                    raise SystemExit(f"RAM used {100 * r:.1f}% > {100 * RAM_ABORT:.0f}%: abort")
            z = {-1: scene.place(-1, *fr["ego_pose"])}
            used = {}
            for j, t in enumerate(fr["tid"]):
                kind = (bool(fr["two_wheeler"][j]), int(fr["cls"][j]))
                key = kind + (used.get(kind, 0),)
                used[kind] = key[2] + 1
                if key not in scene.actors:
                    scene.spawn(key, kind[0], CLASS_RGB[kind[1]])
                z[int(t)] = scene.place(key, *fr["xy_world"][j], fr["yaw_world"][j])
            for key in scene.actors:
                if key != -1 and key[2] >= used.get(key[:2], 0):
                    scene.park(key)
            arr, cam2world = capture(world, cam, q, WARMUP_TICKS if n == 0 else FRAME_TICKS)
            save_frame(arr, cam2world, fr, meta, args.out, z, args.quality)
            fits.append((lane_fit(cmap, fr), fr))
        mem_report("end")
        cam.stop()
        cam.destroy()
        scene.destroy()
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
    finally:
        stop_carla()
    print(f"lane fit, all {len(fits)} frames (target-frames):")
    print_lane_fit([f for fit, _ in fits for f in fit],
                   {k: np.concatenate([fr[k] for _, fr in fits]) for k in ("moving", "tid", "two_wheeler", "cls")})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--frame", type=int, required=True)
    sub.add_parser("render")
    for p in sub.choices.values():
        p.add_argument("--data", default="viz_env/carla_replay/scene16_ego4", help="export_carla_replay.py output")
        p.add_argument("--out", default="viz_env/carla_replay/scene16_ego4_render")
        p.add_argument("--rhi", choices=("opengl", "vulkan"), default="opengl", help="CARLA render backend")
        p.add_argument("--quality", choices=("Low", "Epic"), default="Epic", help="CARLA -quality-level")
    args = ap.parse_args()
    running = busy()
    if running:
        sys.exit("refusing: a run / training / evaluation process or a CARLA server is running:\n  "
                 + "\n  ".join(running))
    {"smoke": cmd_smoke, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
