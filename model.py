
import torch
import torch.nn as nn


class TinyGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        hidden_size=64
    ):
        super().__init__()

        # Convert token IDs into vectors
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        # Learn sequence patterns
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict the next character
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        logits = self.fc(output)
        return logits


if __name__ == "__main__":
    vocab_size = 76

    model = TinyGRU(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=64
    )

    # Test with a batch of 4 sequences, each 50 tokens long
    sample = torch.randint(0, vocab_size, (4, 50))
    output = model(sample)

    parameter_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Model created successfully!")
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Trainable parameters:", parameter_count)
