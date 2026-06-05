import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, num_head * d_k) # For all heads
        self.w_k = nn.Linear(d_model, num_head * d_k)
        self.w_v = nn.Linear(d_model, num_head * d_v)

        self.attn_dropout = nn.Dropout(dropout)

        self.w_o = nn.Linear(num_head * d_v, d_model)

    def forward(self, x):
        # x : [batch_size, seq_len, d_embed]
        batch_size, seq_len, _ = x.shape

        # q, k : [batch_size, seq_len, num_head * d_k]
        # v    : [batch_size, seq_len, num_head * d_v]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, seq_len, self.num_head, self.d_k)
        k = k.view(batch_size, seq_len, self.num_head, self.d_k)
        v = v.view(batch_size, seq_len, self.num_head, self.d_v)

        # q, k : [batch_size, num_head, seq_len, d_k]
        # v    : [batch_size, num_head, seq_len, d_v]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attn_scores : [batch_size, num_head, seq_len, d_k] @ [batch_size, num_head, d_k, seq_len] -> [batch_size, num_head, seq_len, seq_len]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # context : [batch_size, num_head, seq_len, seq_len] @ [batch_size, num_head, seq_len, d_v] -> [batch_size, num_head, seq_len, d_v]
        context = attn_weights @ v
        # context : [batch_size, seq_len, num_head, d_v]
        context = context.transpose(1, 2).contiguous()

        concat_out = context.view(batch_size, seq_len, self.d_model)

        return self.w_o(concat_out)

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_k=64, d_v=64, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, num_head * d_k)
        self.w_k = nn.Linear(d_model, num_head * d_k)
        self.w_v = nn.Linear(d_model, num_head * d_v)

        self.attn_dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(num_head * d_v, d_model)

        # Causal Mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_head, self.d_v).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Filling in -inf will give 0 weight after softmax
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = (attn_weights @ v).transpose(1, 2).contiguous()
        concat_out = context.view(batch_size, seq_len, self.d_model)

        return self.w_o(concat_out)

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, num_head * d_k)
        self.w_k = nn.Linear(d_model, num_head * d_k)
        self.w_v = nn.Linear(d_model, num_head * d_v)

        self.attn_dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(num_head * d_v, d_model)

    def forward(self, x, enc_out):
        batch_size, seq_len_q, _ = x.shape
        _, seq_len_k, _ = enc_out.shape

        # Q from decoder
        q = self.w_q(x).view(batch_size, seq_len_q, self.num_head, self.d_k).transpose(1, 2)

        # K, V from Encoder output
        k = self.w_k(enc_out).view(batch_size, seq_len_k, self.num_head, self.d_k).transpose(1, 2)
        v = self.w_v(enc_out).view(batch_size, seq_len_k, self.num_head, self.d_v).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = (attn_weights @ v).transpose(1, 2).contiguous()
        concat_out = context.view(batch_size, seq_len_q, self.d_model)

        return self.w_o(concat_out)

