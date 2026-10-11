# Decision rules: does observed history improve accuracy?

Written and committed **before any history training was run**. This is the first
step of the accuracy-improvement phase. It uses the **tuning split only**. The test
split is not used for any part of this study.

## Question

Does giving the model the agent's own observed past positions (the last 8 frames)
lower prediction error compared with the same model, on the same samples, without them?

## Input (fixed here, `pilot_dataset.py`, `history_encoder.py`)

- `history_frames=8`: a sample is kept only if the target agent has frames
  `frame-8 .. frame`. Of D's 14,904 samples, this drops 1,512 and keeps 13,392
  (tuning: 1,984 of 2,208). The tuning scenes are unchanged (the same 8 scenes).
- `history` `[8, 2]`: positions at `frame-8 .. frame-1` minus the position at `frame`,
  rotated into the heading at `frame` (the convention of `ground_truth_extractor` and
  `motion_utils`). `history[-1]` equals minus `motion_utils.get_cv_velocity`, which
  `verify_history.py` checks on 50 samples. A parked agent has an all-zero history.
- V2X-Sim records at 5 Hz (secondary sources; the local `sample_data` timestamps are
  frame indices, not real time), so 8 frames is 1.6 s of history, and the horizon of
  30 frames is 6 s.
- Normalization: per-coordinate mean/std over the train split, saved to
  `<run>/history_norm.json`, applied to the returned copy only.
- Encoder: `HistoryEncoder` = GRU(2 → 64), last hidden state → Linear(64 → 128). Its
  output is concatenated with the CAM embedding inside `V2XMessageStream`, whose
  `fuse` layer grows from 256 to 384 inputs. Nothing else in the model changes
  (+37,760 parameters). With history off, the model is D's model: same parameter
  count, and D's checkpoint loads with `strict=True`.

## Runs

All runs use D's flags (`--use-image-cache --use-speed --normalize-cam --drop-cam-xy`,
map on), 12 epochs, seeds 0, 1 and 2:

| Run | Extra flags |
|---|---|
| `Dh_seed{0,1,2}` (control) | `--require-history-frames 8` (same samples, no history input) |
| `H8_seed{0,1,2}` | `--history-frames 8` (GRU encoder) |

Each run is evaluated at `checkpoint_epoch12.pt` on the tuning split
(`eval_scenes.py --run-dir`, writing `eval_<run>_e12.log`), and then
`analyze_seeds.py --rule history --expect-n 1984 --full H8_seed0,H8_seed1,H8_seed2
--nomap Dh_seed0,Dh_seed1,Dh_seed2` is applied (`run_history_seeds.sh`). In that
script's output, "full" means H8 and "no-map" means the control.

## Rule

**History is adopted** only if all three of these hold:

1. seed-mean ALL **ADE1** of H8 is lower than that of the control, **and**
2. seed-mean ALL **minADE6** of H8 is lower than that of the control, **and**
3. seed-mean **ADE1** of H8 is lower than the control's in **≥ 6 of the 8** tuning scenes.

If any of them fails, history is not adopted, and the control stays the baseline.

## Reporting

- The result is reported whatever the outcome, with the per-run and per-scene numbers.
- `diag_history_sensitivity.py` is run on `H8_seed{0,1,2}` and reported. It is
  descriptive only and is not part of the rule.
- The control's numbers are not compared with D's 2,208-sample results. They are on a
  different sample set.

## Not decided here

`--history-encoder transformer` (`TransformerHistoryEncoder`: 2 layers, 4 heads, learned
positional embedding over the 8 steps, mean-pool, Linear → 128) is implemented and
dry-run only. It will be trained and evaluated only after the GRU decision above,
under its own rule, which will be written and committed before those runs.
