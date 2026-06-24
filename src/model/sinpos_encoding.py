import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        matrix = torch.zeros(max_len, d_model)
        
        # Вектор позиций 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Частоты для вычисления синусов и косинусов
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        matrix[:, 0::2] = torch.sin(position * div_term)
        matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', matrix)  # [max_len, d_model]

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        return x + self.pe[segment_ids]