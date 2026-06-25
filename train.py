import torch
import yaml
from src.model.lightning_module import GPTLightningModule

with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = GPTLightningModule(config)

# Настройка логгеров и колбэков
tensorboard_logger, checkpoint_callback = model.setup_loggers_and_callbacks()

# Создаём Trainer
trainer = pl.Trainer(
    max_epochs=config.get('max_epochs', 10),
    accelerator='auto',
    logger=tensorboard_logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=config.get('gradient_clip_val', 1.0),
    log_every_n_steps=10,
)

# trainer.fit(model, train_dataloader, val_dataloader)

print(" Обучение настроено!")