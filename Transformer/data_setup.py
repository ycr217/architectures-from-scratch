import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(texts, vocab_size=10000):
    # 1. Initialize a Byte-Pair Encoding tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # 2. Define special tokens
    # [PAD]: Padding for short sentences
    # [UNK]: Unknown word
    # [SOS]: Start of Sequence
    # [EOS]: End of Sequence
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )

    # 3. Train
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def setup_data_and_tokenizers():
    print("Loading OPUS-100 Chinese-English dataset...")
    # We slice the first 50,000 rows to keep training under 1 hour on an RTX 4090
    opus_dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train[:100000]")

    # Extract the text into separate lists
    en_texts = [item['en'] for item in opus_dataset['translation']]
    zh_texts = [item['zh'] for item in opus_dataset['translation']]

    print("Training English Tokenizer...")
    en_tokenizer = train_tokenizer(en_texts, vocab_size=10000)

    print("Training Chinese Tokenizer...")
    zh_tokenizer = train_tokenizer(zh_texts, vocab_size=10000)

    # Save tokenizers
    os.makedirs("tokenizers", exist_ok=True)
    en_tokenizer.save("tokenizers/en_tokenizer.json")
    zh_tokenizer.save("tokenizers/zh_tokenizer.json")

    print("Tokenizers saved.")

    # Test
    test_sentence = "Hello, how are you?"
    encoded = en_tokenizer.encode(test_sentence)
    print(f"\nTest - English:")
    print(f"Original: {test_sentence}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

    return opus_dataset, en_tokenizer, zh_tokenizer


if __name__ == "__main__":
    dataset, en_tokenizer, zh_tokenizer = setup_data_and_tokenizers()