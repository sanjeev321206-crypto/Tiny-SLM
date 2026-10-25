
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import TinyGRU

# -------------------------------
# 1. Configuration
# -------------------------------
DEVICE = torch.device("cpu")
RESULTS = Path("results")
DATA_FILE = Path("data/prepared_data.json")
BLOCK_SIZE = 64

torch.manual_seed(42)

# -------------------------------
# 2. Load vocabulary and data
# -------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = data["vocab"]
stoi = {ch: i for i, ch in enumerate(vocab)}

# Invert vocabulary mapping
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(vocab)
unk_id = stoi["<UNK>"]

val_data = torch.tensor(
    data["val_ids"], dtype=torch.long
)

# -------------------------------
# 3. Fixed validation evaluation
# -------------------------------
@torch.no_grad()
def evaluate_fixed(model):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Evaluate every next-character target once.
    # Each chunk starts with up to 64 characters.
    for start in range(0, len(val_data) - 1, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, len(val_data) - 1)

        x = val_data[start:end].unsqueeze(0)
        y = val_data[start + 1:end + 1].unsqueeze(0)

        logits = model(x)

        loss_sum = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="sum"
        )

        total_loss += loss_sum.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens

    model.train()

    return avg_loss, math.exp(min(avg_loss, 20))


# -------------------------------
# 4. Plot training curves
# -------------------------------
for size in [64, 128]:
    history_file = RESULTS / f"history_{size}.csv"

    steps = []
    train_losses = []
    val_losses = []

    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="Training loss")
    plt.plot(steps, val_losses, label="Validation loss")

    plt.xlabel("Training steps")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"GRU-{size}: Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = RESULTS / f"learning_curve_{size}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved graph:", filename)

# -------------------------------
# 5. Load and evaluate both models
# -------------------------------
for size in [64, 128]:
    model = TinyGRU(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=size
    ).to(DEVICE)

    state = torch.load(
        RESULTS / f"gru_{size}.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)

    loss, perplexity = evaluate_fixed(model)

    print(f"\nGRU-{size}")
    print("Fixed validation loss:", round(loss, 4))
    print("Fixed validation perplexity:", round(perplexity, 4))

# -------------------------------
# 6. Generate text using GRU-128
# -------------------------------
model = TinyGRU(
    vocab_size=vocab_size,
    embedding_dim=32,
    hidden_size=128
).to(DEVICE)

state = torch.load(
    RESULTS / "gru_128.pt",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state)
model.eval()


@torch.no_grad()
def generate_text(prompt, max_new_chars=400, temperature=0.8):
    # Unknown characters map to UNK
    ids = [stoi.get(ch, unk_id) for ch in prompt]

    if not ids:
        ids = [stoi.get("\n", 0)]

    context = torch.tensor(
        [ids[-BLOCK_SIZE:]],
        dtype=torch.long,
        device=DEVICE
    )

    generated = list(ids)

    for _ in range(max_new_chars):
        context = torch.tensor(
            [generated[-BLOCK_SIZE:]],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(context)
        next_logits = logits[0, -1, :] / temperature

        # Do not generate the special unknown token
        next_logits[unk_id] = float("-inf")

        probabilities = torch.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(
            probabilities, num_samples=1
        ).item()

        generated.append(next_id)

    return "".join(itos[i] for i in generated)


prompt = "Once upon a time"

sample = generate_text(
    prompt,
    max_new_chars=400,
    temperature=0.8
)

print("\n--- GENERATED TEXT ---\n")
print(sample)

with open(
    RESULTS / "generated_story.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(sample)

print("\nSaved generated text to results/generated_story.txt")
print("Evaluation completed!")
