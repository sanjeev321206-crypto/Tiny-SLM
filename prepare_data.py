
from datasets import load_dataset
import json
import random
from pathlib import Path

# Reproducible experiment
random.seed(42)

print("Loading dataset...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1000]"
)

# Split by story, not by character
stories = [item["text"] for item in dataset if item["text"].strip()]
random.shuffle(stories)

split_index = int(len(stories) * 0.9)
train_stories = stories[:split_index]
val_stories = stories[split_index:]

# Build vocabulary using training text only
train_text = "\n".join(train_stories)
val_text = "\n".join(val_stories)

vocab = sorted(set(train_text))

# Handle any validation characters not seen during training
if "<UNK>" not in vocab:
    vocab.append("<UNK>")

stoi = {ch: i for i, ch in enumerate(vocab)}
unk_id = stoi["<UNK>"]

def encode(text):
    return [stoi.get(ch, unk_id) for ch in text]

train_ids = encode(train_text)
val_ids = encode(val_text)

# Save prepared data
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "prepared_data.json", "w", encoding="utf-8") as f:
    json.dump({
        "train_ids": train_ids,
        "val_ids": val_ids,
        "vocab": vocab,
        "train_stories": len(train_stories),
        "val_stories": len(val_stories)
    }, f)

print("\nData preparation complete!")
print("Training stories:", len(train_stories))
print("Validation stories:", len(val_stories))
print("Vocabulary size:", len(vocab))
print("Training characters:", len(train_ids))
print("Validation characters:", len(val_ids))
print("Saved to data/prepared_data.json")
