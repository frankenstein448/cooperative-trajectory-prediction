#!/usr/bin/env bash
# make_bev_figures.sh
# Reproduces every BEV output of render_env_bev.py (tuning split v2, seed 0, epoch 12) with the flags used for the
# thesis and the defense, into viz_env/ of the data checkout ~/hd-maps (old outputs are overwritten):
#
# Usage (from anywhere; works in the data checkout ~/hd-maps, scripts from this checkout):
#   ~/hd-maps-dev/make_bev_figures.sh
#
#   videos (mp4 at 2.5 fps, frames/, 300 dpi stills/)      viz_env/scene<S>_<town>_ego<A>_<A>_vs_<B>/
#     Town05 #22 ego 1  E_V2X | E_EgoOnly    stills 39,     all-modes still 39
#     Town05 #16 ego 4  E_V2X | E_EgoOnly    stills 35, 53, all-modes still 35
#     Town04 #30 ego 3  E_V2X | E_EgoOnly    stills 40,     all-modes still 40
#     Town04 #30 ego 3  E_V2X+RSU | E_V2X    stills 19, 40, all-modes still 40
#   thesis PDFs (--print, 8 files)                          viz_env/thesis_figures/
#     #16 ego 4 frame 35 --no-pred (introduction); #16 ego 4 frame 35 E_V2X | E_EgoOnly (+ all modes);
#     #30 ego 3 frame 40 E_V2X | E_EgoOnly (+ all modes); #30 ego 3 frame 40 E_V2X+RSU | E_V2X (+ all modes);
#     #22 ego 1 frame 39 E_V2X | E_EgoOnly all modes only (appendix)
#   LiDAR blind spot (diag_lidar_behind.py --print)          viz_env/thesis_figures/lidar_blindspot.pdf
#     #30 ego 3 frame 19, LIDAR_TOP top-down (+-25 m), ego box, CAV 2 box: the rear shadow of the ego's body
#
# Predictions come from viz_env/pred/ (computed on the GPU by render_env_bev.py when missing). Refuses to start while
# a run / train / eval process or a GPU compute process is up; prints each command before running it; gpu-test-venv;
# log viz_env/make_bev_figures.log. Stops on the first error.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HOME/hd-maps"
GPU_VENV="$HOME/gpu-test-venv"
V2X_EGO=E_V2X_seed0,E_EgoOnly_seed0
RSU_V2X=E_V2XRSU_seed0,E_V2X_seed0

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
case "${1:-}" in
    "") ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
esac

cd "$DATA"
LOG=viz_env/make_bev_figures.log
mkdir -p viz_env
: > "$LOG"

# Run / train / eval processes (render_env_bev.busy()'s pattern) and GPU compute processes; this script and its
# ancestors are excluded (a calling shell's command line may contain the names).
check_idle() {
    local mine=" " p=$$ found gpu
    while [[ "$p" -gt 1 ]]; do mine+="$p "; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
    found=$(ps -eo pid=,args= | awk -v mine="$mine" '
        index(mine, " " $1 " ") { next }
        $0 ~ /(run_[A-Za-z0-9_]+\.sh|train\.py|eval_[A-Za-z0-9_]+\.py)/ { print }')
    gpu=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
    [[ -n "$gpu" ]] && found+="${found:+$'\n'}GPU compute: $gpu"
    if [[ -n "$found" ]]; then
        echo "refusing to start: something else is running:" >&2
        echo "$found" >&2
        exit 1
    fi
}

# bev <args...>: idle check, command printed, output appended to the log.
bev() {
    check_idle
    local cmd=(nice -n 19 python "$CODE/render_env_bev.py" render "$@")
    echo "+ ${cmd[*]}" | tee -a "$LOG"
    "${cmd[@]}" 2>&1 | tee -a "$LOG"
}

# video <town> <scene> <agent> <runs> <stills> <all-modes stills>: the frames of an earlier render are removed first
# (ffmpeg reads frame_%04d.png until the first gap, so stale higher-numbered frames would end up in the mp4).
video() {
    local town="$1" scene="$2" agent="$3" runs="$4"
    local frames="viz_env/scene${scene}_${town}_ego${agent}_${runs//,/_vs_}/frames"
    echo "+ rm -f $frames/frame_*.png" | tee -a "$LOG"
    rm -f "$frames"/frame_*.png
    bev --scene "$scene" --agent "$agent" --runs "$runs" --stills "$5" --all-modes-stills "$6"
}

source "$GPU_VENV/bin/activate"
echo "BEV figures: data $DATA, code $CODE ($(git -C "$CODE" rev-parse --short HEAD))" | tee -a "$LOG"

echo "=== videos" | tee -a "$LOG"
video Town05 22 1 "$V2X_EGO" 39 39
video Town05 16 4 "$V2X_EGO" 35,53 35
video Town04 30 3 "$V2X_EGO" 40 40
video Town04 30 3 "$RSU_V2X" 19,40 40

echo "=== thesis PDFs" | tee -a "$LOG"
bev --scene 16 --agent 4 --no-pred --print --stills 35
bev --scene 16 --agent 4 --runs "$V2X_EGO" --print --stills 35 --all-modes-stills 35
bev --scene 30 --agent 3 --runs "$V2X_EGO" --print --stills 40 --all-modes-stills 40
bev --scene 30 --agent 3 --runs "$RSU_V2X" --print --stills 40 --all-modes-stills 40
bev --scene 22 --agent 1 --runs "$V2X_EGO" --print --all-modes-stills 39

echo "=== LiDAR blind spot" | tee -a "$LOG"
check_idle
cmd=(nice -n 19 python "$CODE/diag_lidar_behind.py" --print --out viz_env/thesis_figures/lidar_blindspot.pdf)
echo "+ ${cmd[*]}" | tee -a "$LOG"
"${cmd[@]}" 2>&1 | tee -a "$LOG"

echo "=== done:" | tee -a "$LOG"
ls -1 viz_env/scene*_ego*_vs_*/*.mp4 viz_env/scene*_ego*_vs_*/stills/*.png viz_env/thesis_figures/*.pdf | tee -a "$LOG"
