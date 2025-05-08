import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from models.blocks import Mlp, Attention, CrossAttention

############# IMU ENCODER BLOCKS ############# 

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

    def forward(self, x, mask=None):
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
        
        if mask is not None:
            extended_mask = mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(extended_mask == 0, float('-inf'))

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

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TransformerEncoderWithRoPE(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, nhead, mlp_ratio, dropout)
            for _ in range(num_layers)
            ])

    def forward(self, x, masks):
        x = self.input_proj(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, masks)
        return x


############# MODIFIED DECODER BLOCKS ############# 

class MultimodalCrossAttention(nn.Module):

    # cross attention between IMU embedding and unmasked image embedding
    # we need this module because IMU embedding uses 1d RoPE, while image uses 2d RoPE

    def __init__(self,
                 d_model,
                 nhead,
                 rope=None,
                 qkv_bias=False,
                 is_imu1=False, # whether the first input is an imu sequence (for RoPE impl)
                 is_imu2=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.projq = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.projk = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.projv = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.img_rope = rope
        self.imu_rope = RotaryEmbedding(dim=self.head_dim)

        self.is_imu1 = is_imu1
        self.is_imu2 = is_imu2

    def forward(self, x, x_pos, context, context_pos):
        # x is imu data, context is unmasked image
        B, S_x, _ = x.shape
        _, S_context, _ = context.shape
        q = self.projq(x)
        k = self.projq(context)
        v = self.projv(context)

        q = q.view(B, S_x, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S_context, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S_context, self.nhead, self.head_dim).transpose(1, 2)

        if self.is_imu1:
            q = self.imu_rope.rotate_queries_or_keys(q)
        else:
            q = self.img_rope(q, x_pos)

        if self.is_imu2:
            k = self.imu_rope.rotate_queries_or_keys(k)
        else:
            k = self.img_rope(k, context_pos)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, S_x, self.d_model)
        return self.proj_drop(self.out_proj(attn_output))

class IMUDecoderBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_mem=True,
                 rope=None):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

        self.self_attn1 = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        self.self_attn2 = RotarySelfAttention(d_model=dim, nhead=num_heads, attn_drop=drop, proj_drop=drop)
        self.fuse1 = nn.Linear(2*dim, dim)
        self.fuse2 = nn.Linear(2*dim, dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm3 = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm4 = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm5 = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm6 = norm_layer(dim) if norm_mem else nn.Identity()

        # cross attention layers
        # 1: masked image attends to unmasked image
        # 2: masked image -> masked imu sequence
        # 3: masked imu sequence -> unmasked image
        # 4: masked imu sequence -> masked image
        self.cross_attn1 = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        self.cross_attn2 = MultimodalCrossAttention(
                d_model=dim,
                nhead=num_heads,
                rope=rope,
                qkv_bias=qkv_bias,
                is_imu1=False,
                is_imu2=True,
                attn_drop=drop,
                proj_drop=drop
                )
        self.cross_attn3 = MultimodalCrossAttention(
                d_model=dim,
                nhead=num_heads,
                rope=rope,
                qkv_bias=qkv_bias,
                is_imu1=True,
                is_imu2=False,
                attn_drop=drop,
                proj_drop=drop
                )
        self.cross_attn4 = MultimodalCrossAttention(
                d_model=dim,
                nhead=num_heads,
                rope=rope,
                qkv_bias=qkv_bias,
                is_imu1=True,
                is_imu2=False,
                attn_drop=drop,
                proj_drop=drop
                )

    def forward(self, masked_img, masked_img_pos, imu, unmasked_img, unmasked_img_pos):
        # TODO: separate norm objects for each call
        unmasked_img_ = self.norm_y(unmasked_img)
        masked_img = masked_img + self.self_attn1(self.norm1(masked_img), masked_img_pos)
        imu = imu + self.self_attn2(self.norm2(imu))
        masked_img_norm = self.norm3(masked_img)
        imu_norm = self.norm4(imu)

        x1 = self.cross_attn1(masked_img_norm,
                              unmasked_img_,
                              unmasked_img_,
                              masked_img_pos,
                              unmasked_img_pos)
        x2 = self.cross_attn2(masked_img_norm,
                              masked_img_pos,
                              imu_norm,
                              None)
        x_fused = self.fuse1(torch.cat([x1, x2], dim=-1))
        x = masked_img + x_fused
        x = x + self.mlp1(self.norm5(x))

        y1 = self.cross_attn3(imu_norm,
                              None,
                              unmasked_img_,
                              unmasked_img_pos)
        y2 = self.cross_attn4(imu_norm,
                              None,
                              masked_img_norm,
                              masked_img_pos)
        y_fused = self.fuse2(torch.cat([y1, y2], dim=-1))
        y = imu + y_fused
        y = y + self.mlp2(self.norm6(y))

        return x, y, masked_img
