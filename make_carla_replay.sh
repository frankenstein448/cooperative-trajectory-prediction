#!/usr/bin/env bash
# make_carla_replay.sh
# Reproduces the CARLA replay video of one tuning-split scene / ego end to end (visualisation only: recorded V2X-Sim
# data replayed in CARLA 0.9.15, the cached seed-0 epoch-12 predictions of E_V2X and E_EgoOnly drawn over it).
#
# Usage (from anywhere; works in the data checkout ~/hd-maps, scripts from this checkout):
#   ~/hd-maps-dev/make_carla_replay.sh [--scene 16] [--agent 4] [--still-frame 35]
#
#   step 0  predictions cache viz_env/pred/<run>_scene<S>_ego<A>.npz via render_env_bev.py (only if missing)
#                                                                                    gpu-test-venv
#   step 1  export_carla_replay.py   -> viz_env/carla_replay/scene<S>_ego<A>/        gpu-test-venv
#   step 2  carla_replay.py render   -> viz_env/carla_replay/scene<S>_ego<A>_render/ carla-venv (starts / stops CARLA)
#   step 3  overlay_carla_replay.py  -> viz_env/carla_replay/scene<S>_ego<A>_overlay/ (frames/, mp4 at 2.5 fps,
#                                       overlay_frame<NNN>_300dpi.png)               gpu-test-venv
#
# Each step: refuses to start while a run / train / eval process, another CARLA server or any GPU compute process is
# up; prints its exact command; logs to viz_env/carla_replay/<step>_scene<S>_ego<A>.log. Stops on the first error.
set -euo pipefail

SCENE=16
AGENT=4
STILL=35
RUNS=E_V2X_seed0,E_EgoOnly_seed0
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HOME/hd-maps"
GPU_VENV="$HOME/gpu-test-venv"
CARLA_VENV="$HOME/carla-venv"

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --still-frame) STILL="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

cd "$DATA"
OUT=viz_env/carla_replay
NAME="scene${SCENE}_ego${AGENT}"
mkdir -p "$OUT"

# Run / train / eval processes (same pattern as the Python scripts' busy()), CARLA servers and GPU compute
# processes; this script and its ancestors are excluded (a calling shell's command line may contain the names).
check_idle() {
    local mine=" " p=$$ found gpu
    while [[ "$p" -gt 1 ]]; do mine+="$p "; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
    found=$(ps -eo pid=,comm=,args= | awk -v mine="$mine" '
        index(mine, " " $1 " ") { next }
        $2 ~ /^CarlaUE4/ || $0 ~ /(run_[A-Za-z0-9_]+\.sh|train\.py|eval_[A-Za-z0-9_]+\.py)/ { print }')
    gpu=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
    [[ -n "$gpu" ]] && found+="${found:+$'\n'}GPU compute: $gpu"
    if [[ -n "$found" ]]; then
        echo "refusing to start $1: something else is running:" >&2
        echo "$found" >&2
        exit 1
    fi
}

# step <name> <venv> <command...>: idle check, venv in a subshell, command printed, output to the step's log.
step() {
    local name="$1" venv="$2"; shift 2
    local log="$OUT/${name}_${NAME}.log"
    check_idle "$name"
    echo "=== $name ($(basename "$venv"), log $log)"
    echo "+ $*"
    ( source "$venv/bin/activate" && "$@" ) 2>&1 | tee "$log"
}

echo "CARLA replay: scene $SCENE, ego CAV $AGENT, still frame $STILL; data $DATA, code $CODE"

# step 0: the export reads these (render_env_bev.load_preds checks them against the runs' eval_env_tuning npz)
missing=0
for r in ${RUNS//,/ }; do
    [[ -f "viz_env/pred/${r}_scene${SCENE}_ego${AGENT}.npz" ]] || missing=1
done
if [[ $missing -eq 1 ]]; then
    step pred "$GPU_VENV" nice -n 19 python "$CODE/render_env_bev.py" render --scene "$SCENE" --agent "$AGENT" \
        --runs "$RUNS" --frames 1 --no-video
else
    echo "=== pred: viz_env/pred/{${RUNS}}_scene${SCENE}_ego${AGENT}.npz present, skipped"
fi

# step 1
step export "$GPU_VENV" nice -n 19 python "$CODE/export_carla_replay.py" --scene "$SCENE" --agent "$AGENT" \
    --runs "$RUNS"
STILL_NPZ=$(printf "%s/%s/frame_%03d.npz" "$OUT" "$NAME" "$STILL")
[[ -f "$STILL_NPZ" ]] || { echo "--still-frame $STILL: $STILL_NPZ was not exported" >&2; exit 1; }

# step 2
step render "$CARLA_VENV" python "$CODE/carla_replay.py" render --data "$OUT/$NAME" --out "$OUT/${NAME}_render" \
    --rhi opengl --quality Epic

# step 3
step overlay "$GPU_VENV" nice -n 19 python "$CODE/overlay_carla_replay.py" --all --stills "$STILL" \
    --models E_V2X,E_EgoOnly --data "$OUT/$NAME" --render "$OUT/${NAME}_render" --out "$OUT/${NAME}_overlay"

echo "=== done:"
ls -1 "$OUT/${NAME}_overlay"/*.mp4 "$OUT/${NAME}_overlay"/overlay_frame*_300dpi.png
