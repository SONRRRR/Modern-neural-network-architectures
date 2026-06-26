"""Model modules."""

import torch
from src.model.gpt_model import GPTModel


def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = GPTModel(config)
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.get('pad_token_id', 0))
    self.save_hyperparameters(config)
    
    # Добавь эту строку:
    self.gradient_clip_val = config.get('gradient_clip_val', 1.0)