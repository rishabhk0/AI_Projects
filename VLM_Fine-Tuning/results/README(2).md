# Results

## Zero-shot baseline

Idefics3-8B, no adapter, evaluated with `model.disable_adapter()` on 50 held-out ScienceQA image questions: **84.0%**.

## A bug that shaped every training run below

`format_example()` in the notebook this project came from set `labels = input_ids.clone()` with no masking, so training loss was computed over the entire sequence: image tokens, the question, the choices, and the answer, not just the answer. `train.py` in this repo fixes that (loss masked to answer tokens only via a `-100` label mask) - but that fix was written after all four runs below had already been executed, so none of the numbers here reflect it. The corrected version has not been run yet.

One run (run4) was logged with an MLflow param claiming `"label_masking": "fixed"`. That param was wrong. The code that actually ran was identical to run1's. Worth being direct about that rather than letting a logged tag stand in for what the code actually did.

## Runs

| Run | LoRA rank | Train samples | Accuracy | vs. baseline | Gate |
|---|---|---|---|---|---|
| Zero-shot | n/a | n/a | 84.0% | - | baseline |
| run1 | 16 | 300 | 66.0% | -18.0 | blocked |
| run2 | 4 | 300 | 72.0% | -12.0 (vs zero-shot) | would be blocked against zero-shot; was compared against run1 at the time |
| run3 | 2 | 20 | 84.0% | 0.0 | approved - matches zero-shot exactly, consistent with ~2 optimizer steps changing almost nothing |
| run4 | 16 | 300 | 68.0% | -16.0 | blocked |

Full loss curve for run4 is in `run4_loss_curve.csv`.

## What's actually true here

Run1 and run4 used the same buggy config and landed close together (66% and 68%), which is a reasonable outcome for two runs of the same code with different random shuffling. Run4's loss dropped to near zero by the end of training. Given the unmasked-loss bug, that's expected regardless of whether the model got better or worse at the actual task, since the objective includes learning to reproduce prompt and image tokens that are deterministic given the input. So the loss curve isn't strong evidence that the model specifically overfit on answers. What's solid is the accuracy number itself: both buggy rank-16/300 runs scored meaningfully below zero-shot on a held-out set, and the deployment gate caught that both times.

Run3 is the cleanest result in the batch, in a roundabout way. Rank 2 on 20 examples with gradient accumulation of 8 means roughly 2 optimizer steps total, next to no training. It scored exactly at the zero-shot baseline, which is exactly what you'd expect if training barely touched the weights. That's a useful sanity check on the eval pipeline itself, even though it wasn't the finding the run was originally designed to produce.

## What's still open

The corrected `train.py` (proper label masking) hasn't been run. Rerunning rank 16 on 300 examples with the fix would answer the actual question this project set out to answer: does QLoRA fine-tuning help on this task at all, once the loss is computed correctly. Until that run happens, the honest claim is "built and validated a training pipeline with a working eval gate," not "fine-tuning improved or degraded accuracy," since the numbers above are entangled with a bug we already found and haven't retested past.
