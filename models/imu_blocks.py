import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from models.blocks import Mlp

class RotarySelfAttention(nn.Module):
    # we need RoPE for imu data (which has 1d position) but currently implemented blocks
    # use 2d positions from the patch embeddings, need to reimplement encoder blocks

    def __init__(self, d_model, nhead, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead 
        self.head_dim= d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, S, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, S, nhead, head_dim] -> [B, nhead, S, head_dim]
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # apply RoPE to q and k
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)	

        # attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, S, head_dim]

        # merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.d_model)
        return self.proj_drop(self.out_proj(attn_output))


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4., drop=0.):
        super().__init__()
        self.self_attn = RotarySelfAttention(d_model, nhead, attn_drop=drop, proj_drop=drop)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, drop=drop)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TransformerEncoderWithRoPE(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.Sequential(*[
            TransformerEncoderLayerWithRoPE(d_model, nhead, mlp_ratio, dropout)
            for _ in range(num_layers)
            ])

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder_layers(x)
        return x

