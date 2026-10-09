"""
check_zero_speed.py
Metadata-only check (no sensor loads): for every (agent_id, scene), over all
consecutive frame pairs (f-1, f) where both exist, the fraction with exactly
zero displacement and the longest consecutive run of zeros. Alternating
0/nonzero would point at duplicated poses (a bug); long runs point at
genuinely stopped agents.
"""
import math
from pilot_dataset import ALL_SCENES, CAMERA_COMPATIBLE_AGENTS, SCENE_TO_TOWN, _available_frames
from v2x_message_stream import V2XMetadataLoader
from motion_utils import get_cv_velocity

loader = V2XMetadataLoader()

rows = []  # (town, scene, agent_id, n_pairs, n_zero, frac_zero, longest_run, n_transitions, transition_rate)
for scene in ALL_SCENES:
    town = SCENE_TO_TOWN[scene]
    for agent_id in CAMERA_COMPATIBLE_AGENTS:
        frames = sorted(_available_frames(agent_id, scene))
        frame_set = set(frames)

        zeros_seq = []
        for f in frames:
            if (f - 1) not in frame_set:
                continue
            ok, vx, vy = get_cv_velocity(loader, agent_id, scene, f)
            assert ok, f"frame-1 present in frame_set but get_cv_velocity failed for ({agent_id},{scene},{f})"
            zeros_seq.append(math.hypot(vx, vy) == 0.0)

        n_pairs = len(zeros_seq)
        if n_pairs == 0:
            continue
        n_zero = sum(zeros_seq)
        frac_zero = n_zero / n_pairs

        longest_run = 0
        cur_run = 0
        for z in zeros_seq:
            cur_run = cur_run + 1 if z else 0
            longest_run = max(longest_run, cur_run)

        n_transitions = sum(1 for i in range(1, n_pairs) if zeros_seq[i] != zeros_seq[i - 1])
        transition_rate = n_transitions / (n_pairs - 1) if n_pairs > 1 else 0.0

        rows.append((town, scene, agent_id, n_pairs, n_zero, frac_zero, longest_run, n_transitions, transition_rate))

total_pairs = sum(r[3] for r in rows)
total_zero = sum(r[4] for r in rows)
print(f"Overall: {total_zero}/{total_pairs} consecutive pairs are exact-zero displacement "
      f"({100 * total_zero / total_pairs:.1f}%)")
worst = max(rows, key=lambda r: r[6])
print(f"Longest zero-run overall: {worst[6]} frames (town={worst[0]} scene={worst[1]} agent={worst[2]})")

print("\nPer agent (summed across all scenes):")
for agent_id in CAMERA_COMPATIBLE_AGENTS:
    agent_rows = [r for r in rows if r[2] == agent_id]
    n_pairs = sum(r[3] for r in agent_rows)
    n_zero = sum(r[4] for r in agent_rows)
    max_run = max((r[6] for r in agent_rows), default=0)
    mean_transition_rate = sum(r[8] for r in agent_rows) / len(agent_rows) if agent_rows else 0.0
    print(f"  agent {agent_id}: {n_zero}/{n_pairs} zero ({100 * n_zero / n_pairs:.1f}%), "
          f"longest run={max_run}, mean transition_rate={mean_transition_rate:.3f}")

print("\nSuspicious (agent, scene) combos -- high transition_rate (>0.7) with enough pairs to matter (n_pairs>=20):")
suspicious = [r for r in rows if r[8] > 0.7 and r[3] >= 20]
if suspicious:
    for r in suspicious:
        print(f"  town={r[0]} scene={r[1]} agent={r[2]}: n_pairs={r[3]} frac_zero={r[5]:.2f} "
              f"longest_run={r[6]} transition_rate={r[8]:.2f}")
else:
    print("  none found")

print("\nTop 10 by longest zero-run:")
for r in sorted(rows, key=lambda r: -r[6])[:10]:
    print(f"  town={r[0]} scene={r[1]} agent={r[2]}: longest_run={r[6]}  n_pairs={r[3]}  frac_zero={r[5]:.2f}")
