import torch.nn as nn
import math

from blocks import PositionalEncoding, EncoderTransformerBlock, DecoderTransformerBlock

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_head=8, num_layers=6, d_ffn=2048, d_k=64, d_v=64, dropout=0.1, max_seq_len=1024):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder Stack
        self.encoder_blocks = nn.ModuleList([
            EncoderTransformerBlock(d_model, num_head, d_ffn, d_k, d_v, dropout)
            for _ in range(num_layers)
        ])

        # Decoder Stack
        self.decoder_blocks = nn.ModuleList([
            DecoderTransformerBlock(d_model, num_head, d_ffn, d_k, d_v, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src):
        # scale base on original paper
        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src = self.pos_encoding(src)

        for block in self.encoder_blocks:
            src = block(src)

        return src

    def decode(self, tgt, enc_out):
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt = self.pos_encoding(tgt)

        for block in self.decoder_blocks:
            tgt = block(tgt, enc_out)

        return tgt

    def forward(self, src, tgt):
        enc_out = self.encode(src)
        dec_out = self.decode(tgt, enc_out)
        logits = self.generator(dec_out)

        return logits