import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.transformer import Transformer
from data_setup import setup_data_and_tokenizers
from data_loader import get_dataloader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, en_tokenizer, zh_tokenizer = setup_data_and_tokenizers()

    dataloader = get_dataloader(dataset, src_tokenizer=en_tokenizer, tgt_tokenizer=zh_tokenizer, batch_size=64)

    vocab_size = 10000
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        num_head=8,
        num_layers=6,
        max_seq_len=128
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 5
    model.train()

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")

        total_loss = 0

        for batch in loop:
            # Move everything to the GPU
            enc_input, dec_input, labels = batch
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(enc_input, dec_input)

            # Reshape for CrossEntropyLoss
            # Logits shape: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # Labels shape: [batch_size, seq_len] -> [batch_size * seq_len]
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

        # Save the trained weights
    torch.save(model.state_dict(), "transformer_en_zh.pth")
    print("Training complete! Model saved.")


if __name__ == "__main__":
    train()