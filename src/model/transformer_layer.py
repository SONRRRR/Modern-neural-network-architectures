import torch
import torch.nn as nn
from .attention import MultiHeadMaskedAttention
from .ffn import FeedForward

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadMaskedAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, segment_ids=None):
        # z1 = LayerNorm(x + Attention(x))
        attn_out = self.attention(x, segment_ids)
        x = self.norm1(x + self.dropout(attn_out))
        
        # z2 = LayerNorm(z1 + FFN(z1))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x