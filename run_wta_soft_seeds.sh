#!/bin/bash
# runs/DECISION_RULES_LOSS.md, fallback section: W_seed{0..4} (D + --loss wta_soft, tau 1.0 m) vs the
# existing D_seed{0..4}, 12 epochs, each evaluated at epoch 12 on the tuning split; then the pre-registered rule.
#   1. verify_loss.py (CPU, random tensors) must pass.
#   2. W_seed0..4: train, eval_scenes.py at checkpoint_epoch12.pt (ranking metrics) -> eval_W_seed*_e12.log.
#   3. analyze_seeds.py --rule loss --candidate wta_soft against D's existing logs (nothing of D is rerun).
set -euo pipefail
source ~/gpu-test-venv/bin/activate
ulimit -n 65535
FLAGS="--epochs 12 --use-image-cache --use-speed --normalize-cam --drop-cam-xy"

# D's logs used by the rule must already exist
for f in eval_D_seed{0,1,2}_e12_rank.log eval_D_seed{3,4}_e12.log; do
  [ -f "$f" ] || { echo "missing $f -- stopping"; exit 1; }
done

CUDA_VISIBLE_DEVICES="" python -u verify_loss.py 2>&1 | tee verify_loss_wta_soft.log

for s in 0 1 2 3 4; do
  name=W_seed$s
  python -u train.py $FLAGS --seed $s --loss wta_soft --out-dir runs/$name 2>&1 | tee train_$name.log
  python -u eval_scenes.py --run-dir runs/$name --ckpt runs/$name/checkpoint_epoch12.pt 2>&1 | tee eval_${name}_e12.log
done

python analyze_seeds.py --rule loss --candidate wta_soft --expect-n 2208 \
  --full W_seed0,W_seed1,W_seed2,W_seed3,W_seed4 \
  --nomap D_seed0:e12_rank,D_seed1:e12_rank,D_seed2:e12_rank,D_seed3,D_seed4 2>&1 | tee analyze_seeds_wta_soft.log
echo "WTA_SOFT SEED RUNS DONE"
