import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, max_seq_len=128):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

        self.pad_id = 0
        self.sos_id = 2
        self.eos_id = 3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pair = self.dataset[idx]['translation']

        # English to Chinese
        src_text = pair['en']
        tgt_text = pair['zh']

        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # Truncate if the sentence is too long (leave room for SOS and EOS)
        src_ids = src_ids[:self.max_seq_len - 2]
        tgt_ids = tgt_ids[:self.max_seq_len - 1]

        # Encoder Input: [SOS] + Source Text + [EOS]
        enc_input = [self.sos_id] + src_ids + [self.eos_id]

        # Decoder Input: [SOS] + Target Text
        # (The decoder predicts the next word, so it starts with SOS and doesn't see EOS yet)
        dec_input = [self.sos_id] + tgt_ids

        # Label: Target Text + [EOS]
        # (This is exactly what the decoder is trying to predict)
        label = tgt_ids + [self.eos_id]

        return {
            "enc_input": enc_input,
            "dec_input": dec_input,
            "label": label
        }


def collate_fn(batch):
    """
    This function takes a batch of varied-length sentences and pads them
    with [PAD] (0) so they all form a perfect rectangle (matrix).
    """
    pad_id = 0

    enc_inputs = [torch.tensor(item["enc_input"]) for item in batch]
    dec_inputs = [torch.tensor(item["dec_input"]) for item in batch]
    labels = [torch.tensor(item["label"]) for item in batch]

    # Pad sequences to the longest sequence *in this specific batch*
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=pad_id)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id)

    return enc_inputs, dec_inputs, labels

def get_dataloader(dataset, src_tokenizer, tgt_tokenizer, batch_size=64):
    train_dataset = TranslationDataset(dataset, src_tokenizer, tgt_tokenizer)
    # shuffle=True is critical so the model doesn't memorize the order of the dataset
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)