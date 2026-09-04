"""
Data loading for ScienceQA image-based questions.

ScienceQA mixes text-only and image-based questions. Only the image-bearing
ones exercise the vision side of the model, so this filters down to those
before any splitting into train/val/test.
"""

from datasets import load_dataset


def load_scienceqa_image_splits(train_n: int = 300, val_n: int = 50, test_n: int = 50):
    raw = load_dataset("derek-thomas/ScienceQA")

    def has_image(example):
        return example["image"] is not None

    train_full = raw["train"].filter(has_image)
    val_full = raw["validation"].filter(has_image)
    test_full = raw["test"].filter(has_image)

    train_data = train_full.select(range(min(train_n, len(train_full))))
    eval_data = val_full.select(range(min(val_n, len(val_full))))
    test_data = test_full.select(range(min(test_n, len(test_full))))

    return train_data, eval_data, test_data, train_full


if __name__ == "__main__":
    train_data, eval_data, test_data, _ = load_scienceqa_image_splits()
    print(f"Train: {len(train_data)}, Val: {len(eval_data)}, Test: {len(test_data)}")
    print(f"\nSample question: {train_data[0]['question']}")
    print(f"Choices: {train_data[0]['choices']}")
    print(f"Answer index: {train_data[0]['answer']}")
