# VLM Fine-Tuning Pipeline

QLoRA fine-tuning of Idefics3-8B on ScienceQA's image-based questions, with MLflow experiment tracking and an automated deployment gate that blocks a model version if held-out accuracy drops more than 2 points versus baseline.

Built as a companion piece to two other portfolio projects (a grounded RAG chatbot and a claim-level hallucination detector) after a conversation about what actually gets checked in the Berlin ML job market: not whether you can train a model, but whether you can ship one and know when it's gotten worse.

## What's here

- `data.py` - loads ScienceQA, filters to the image-bearing subset, splits train/val/test
- `train.py` - QLoRA fine-tuning with MLflow tracking, correct label masking (loss on answer tokens only)
- `eval.py` - held-out evaluation plus the deployment gate logic
- `results/` - the actual runs from development, including a labeling bug that was caught and is documented honestly rather than papered over

## The honest part

Read `results/README.md` before trusting any number in this repo. Short version: the training code used during development had a labeling bug (loss computed over the full sequence instead of just the answer), which was caught, fixed in `train.py`, but not yet re-run. The results directory reports what actually happened, including the bug, rather than a cleaned-up story.

## Setup

```bash
pip install -r requirements.txt
python train.py --lora-rank 16 --train-n 300 --run-name my-run
```

This needs a GPU with at least ~15GB VRAM (developed and tested on a Colab T4). `do_image_splitting=False` and gradient checkpointing are both required to fit an 8B VLM in 4-bit on that hardware - see the comments in `train.py` for why.

## Viewing MLflow results

```bash
mlflow ui
```

Then open `localhost:5000` to see logged params, loss curves, and the gate status per run.

## What this actually demonstrates

Not a strong fine-tuned model - the honest numbers in `results/` don't show one yet. What it demonstrates is a full pipeline: QLoRA fine-tuning that runs on consumer GPU hardware, experiment tracking with real config and metrics, an eval gate that compares against baseline and blocks regressions, and - maybe most relevant to an interviewer - catching and correctly diagnosing a real bug in the training loop rather than shipping numbers that looked fine but weren't measuring what they claimed to.
