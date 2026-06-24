import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Линейные слои для Q, K, V и выходной
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, segment_ids=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Проекции
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if segment_ids is not None:
            same_seq = (segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2))
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
            causal = causal.unsqueeze(0).expand(batch_size, -1, -1)
            not_pad = (segment_ids != 0).unsqueeze(-1) & (segment_ids != 0).unsqueeze(-2)
            
            mask = same_seq & causal & not_pad
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(out)