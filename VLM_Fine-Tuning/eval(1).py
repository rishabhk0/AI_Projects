"""
Evaluation harness for the ScienceQA VLM pipeline. Scores a model against
a held-out set and applies a deployment gate: block if accuracy drops more
than ACCURACY_DROP_THRESHOLD points versus a baseline.

Usage:
    python eval.py  # requires a trained `model` and `processor` in scope,
                     # typically called from a notebook or another script
"""

import torch
from tqdm import tqdm

ACCURACY_DROP_THRESHOLD = 2.0  # percentage points


def evaluate_model(model, processor, dataset, max_new_tokens: int = 5) -> float:
    model.eval()
    correct = 0
    for example in tqdm(dataset, desc="Evaluating"):
        choices_str = ", ".join(f"{i}: {c}" for i, c in enumerate(example["choices"]))
        prompt = f"Question: {example['question']}\nChoices: {choices_str}\nAnswer with the choice number."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[example["image"]], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        predicted = "".join(c for c in generated if c.isdigit())[:1]
        if predicted == str(example["answer"]):
            correct += 1

    return correct / len(dataset) * 100


def check_deployment_gate(current_accuracy: float, baseline_accuracy: float) -> dict:
    drop = baseline_accuracy - current_accuracy
    blocked = drop > ACCURACY_DROP_THRESHOLD
    return {
        "current_accuracy": current_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "change_points": -drop,
        "blocked": blocked,
        "status": "blocked" if blocked else "approved",
    }


def zero_shot_baseline(model, processor, test_data) -> float:
    """Evaluate the base model with its LoRA adapter disabled, so it can
    serve as a baseline the fine-tuned version is compared against."""
    with model.disable_adapter():
        return evaluate_model(model, processor, test_data)
