"""
QLoRA fine-tuning of Idefics3-8B on ScienceQA (image-bearing subset), with
MLflow tracking of config and loss.

IMPORTANT — read before trusting any numbers in results/: the training runs
recorded in results/ were executed with a labeling bug (loss computed over
the full sequence — image tokens, prompt, and answer — instead of the answer
tokens only). format_example() below is the corrected version. It has NOT
yet been run; see results/README.md for exactly which numbers come from the
buggy runs and why they're still reported.

Usage:
    python train.py --lora-rank 16 --train-n 300 --run-name my-run
"""

import argparse
import gc

import torch
import mlflow
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data import load_scienceqa_image_splits

MODEL_ID = "HuggingFaceM4/Idefics3-8B-Llama3"


def load_model_and_processor(lora_rank: int, lora_alpha: int, lora_dropout: float = 0.05):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # do_image_splitting=False matters on a single T4 — Idefics3 tiles each
    # image into several high-res crops by default, which multiplies the
    # per-image sequence length and was the direct cause of an early OOM
    # during development.
    processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def format_example(example, processor):
    """Builds a training example with the loss masked to answer tokens only.

    This is the corrected version. The runs in results/ used a version that
    set labels = input_ids.clone() with no masking, so loss was computed
    over the prompt and image tokens too — see results/README.md.
    """
    choices_str = ", ".join(f"{i}: {c}" for i, c in enumerate(example["choices"]))
    prompt = f"Question: {example['question']}\nChoices: {choices_str}\nAnswer with the choice number."
    answer = str(example["answer"])

    prompt_only = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    full_convo = prompt_only + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    prompt_text = processor.apply_chat_template(prompt_only, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_convo, add_generation_prompt=False)

    prompt_len = processor(
        text=prompt_text, images=[example["image"]], return_tensors="pt"
    )["input_ids"].shape[1]
    inputs = processor(text=full_text, images=[example["image"]], return_tensors="pt")

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # mask everything except the answer tokens
    inputs["labels"] = labels
    return inputs


def train(lora_rank: int, lora_alpha: int, train_n: int, learning_rate: float,
          grad_accum_steps: int, run_name: str, adapter_out: str = "lora_adapter"):
    model, processor = load_model_and_processor(lora_rank, lora_alpha)
    train_data, _, _, _ = load_scienceqa_image_splits(train_n=train_n)

    def collate_fn(batch):
        return format_example(batch[0], processor)

    train_loader = DataLoader(train_data, batch_size=1, collate_fn=collate_fn, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("vlm-scienceqa-finetuning")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_id": MODEL_ID,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "grad_accum_steps": grad_accum_steps,
            "train_samples": train_n,
            "label_masking": "correct - loss computed on answer tokens only",
        })

        model.train()
        step = 0
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            step += 1
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                mlflow.log_metric("train_loss", running_loss / grad_accum_steps, step=step)
                running_loss = 0.0
            if step % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        model.save_pretrained(adapter_out)
        mlflow.log_artifacts(adapter_out, artifact_path="lora_adapter")

    print(f"Training complete. Adapter saved to {adapter_out}.")
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--train-n", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="qlora-run")
    args = parser.parse_args()

    train(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        train_n=args.train_n,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        run_name=args.run_name,
    )
