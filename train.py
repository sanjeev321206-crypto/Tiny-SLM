
import json
import csv
import time
import math
from pathlib import Path

import torch
import torch.nn as nn

from model import TinyGRU

# -------------------------------
# 1. Configuration
# -------------------------------
torch.manual_seed(42)

DEVICE = torch.device("cpu")

BATCH_SIZE = 32
BLOCK_SIZE = 64
MAX_STEPS = 300
EVAL_INTERVAL = 50
LEARNING_RATE = 0.003

DATA_FILE = Path("data/prepared_data.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Training device:", DEVICE)

# -------------------------------
# 2. Load prepared data
# -------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = torch.tensor(
    data["train_ids"], dtype=torch.long
)
val_data = torch.tensor(
    data["val_ids"], dtype=torch.long
)

vocab_size = len(data["vocab"])

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Vocabulary size:", vocab_size)

# -------------------------------
# 3. Create random batches
# -------------------------------
def get_batch(source):
    # Random starting positions
    starts = torch.randint(
        0,
        len(source) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + BLOCK_SIZE]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1]
        for i in starts.tolist()
    ])

    return x.to(DEVICE), y.to(DEVICE)


# -------------------------------
# 4. Evaluate loss
# -------------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = []

        for _ in range(10):
            x, y = get_batch(source)
            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results


# -------------------------------
# 5. Train one model
# -------------------------------
def train_model(hidden_size):
    print("\n" + "=" * 45)
    print("Training GRU with hidden size:", hidden_size)
    print("=" * 45)

    model = TinyGRU(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=hidden_size
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        model.train()

        x, y = get_batch(train_data)
        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # Helps prevent unusually large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0:
            scores = estimate_loss(model)

            history.append({
                "step": step,
                "train_loss": scores["train"],
                "val_loss": scores["val"]
            })

            print(
                f"Step {step:3d}/{MAX_STEPS} | "
                f"Train loss: {scores['train']:.4f} | "
                f"Validation loss: {scores['val']:.4f}"
            )

    elapsed = time.time() - start_time

    # Calculate final metrics
    final_scores = estimate_loss(model)

    result = {
        "hidden_size": hidden_size,
        "parameters": parameter_count,
        "train_loss": final_scores["train"],
        "val_loss": final_scores["val"],
        "val_perplexity": math.exp(
            min(final_scores["val"], 20)
        ),
        "training_seconds": round(elapsed, 2)
    }

    # Save trained weights
    torch.save(
        model.state_dict(),
        RESULTS_DIR / f"gru_{hidden_size}.pt"
    )

    # Save loss history
    with open(
        RESULTS_DIR / f"history_{hidden_size}.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "train_loss", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nModel finished!")
    print("Trainable parameters:", parameter_count)
    print("Final validation loss:", round(final_scores["val"], 4))
    print("Validation perplexity:", round(result["val_perplexity"], 2))
    print("Training time:", elapsed, "seconds")

    return result


# -------------------------------
# 6. Train two model sizes
# -------------------------------
results = []

for size in [64, 128]:
    results.append(train_model(size))

# -------------------------------
# 7. Save comparison table
# -------------------------------
comparison_file = RESULTS_DIR / "comparison.csv"

with open(
    comparison_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 45)
print("FINAL MODEL COMPARISON")
print("=" * 45)

for result in results:
    print(result)

print("\nComparison saved to:", comparison_file)
print("Training completed!")
