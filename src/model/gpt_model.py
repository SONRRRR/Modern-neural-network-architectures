import torch.nn as nn
from .sinpos_encoding import SinusoidalPositionalEncoding
from .transformer_layer import TransformerLayer
from .lm_head import LMHead

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = nn.Embedding(
            config['vocab_size'], 
            config['d_model'],
            padding_idx=config.get('pad_token_id', 0)
        )
        
        self.pos_encoding = SinusoidalPositionalEncoding(
            config['d_model'],
            config.get('max_len', 5000)
        )
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                config['d_model'],
                config['n_heads'],
                config['d_ff'],
                config.get('dropout', 0.1)
            )
            for _ in range(config['n_layers'])
        ])
        
        self.lm_head = LMHead(config['d_model'], config['vocab_size'])
    
    def forward(self, input_ids, segment_ids=None):
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x, segment_ids)
        
        for layer in self.layers:
            x = layer(x, segment_ids)
        
        return self.lm_head(x)