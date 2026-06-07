import torch
import torch.nn as nn
import math

from attention import MultiHeadAttention, CausalMultiHeadAttention, CrossMultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # [max_seq_length, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension [max_seq_len, d_model] -> [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape : [batch_size, seq_len, d_model]
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ffn = 2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class EncoderTransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ffn=2048, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.multi_head_attn = MultiHeadAttention(d_model, num_head, d_k, d_v)
        self.feed_forward = FeedForwardNetwork(d_model, d_ffn)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x

        attn_out = self.multi_head_attn(x)
        attn_out = self.residual_dropout1(attn_out)

        x = self.ln1(res + attn_out)

        res = x

        ffn_out = self.feed_forward(x)
        ffn_out = self.residual_dropout2(ffn_out)

        x = self.ln2(res + ffn_out)

        return x


class DecoderTransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ffn=2048, d_k=64, d_v=64, dropout=0.1, max_seq_len=1024):
        super().__init__()

        self.masked_multi_head_attn = CausalMultiHeadAttention(d_model, num_head, d_k, d_v, dropout, max_seq_len)
        self.cross_multi_head_attn = CrossMultiHeadAttention(d_model, num_head, d_k, d_v, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ffn, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)
        self.residual_dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        # 1. Masked Self-Attention
        res = x
        masked_attn_out = self.masked_multi_head_attn(x)
        masked_attn_out = self.residual_dropout1(masked_attn_out)
        x = self.ln1(res + masked_attn_out)

        # 2. Cross-Attention
        res = x
        cross_attn_out = self.cross_multi_head_attn(x, enc_out)
        cross_attn_out = self.residual_dropout2(cross_attn_out)
        x = self.ln2(res + cross_attn_out)

        # 3. Feed Forward Network
        res = x
        ffn_out = self.feed_forward(x)
        ffn_out = self.residual_dropout3(ffn_out)
        x = self.ln3(res + ffn_out)

        return x