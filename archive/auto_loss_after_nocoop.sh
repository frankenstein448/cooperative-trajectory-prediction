#!/bin/bash
set -euo pipefail
cd ~/hd-maps
echo "Waiting for NoCoop to finish (analyze_seeds_NoCoop.log with VERDICT)..."
until grep -q "VERDICT" analyze_seeds_NoCoop.log 2>/dev/null; do sleep 60; done
echo "NoCoop done at $(date). Merging loss commit c09a06d..."
git merge --no-edit c09a06d
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
pkill -f CarlaUE4 || true
python verify_loss.py 2>&1 | tee verify_loss_main.log
python -u train.py --seed 0 --epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy \
  --loss laplace_mixture --out-dir /tmp/dry_L --dry-run 2>&1 | tee dryrun_L.log
./run_loss_seeds.sh
echo "LOSS STUDY DONE at $(date)"
