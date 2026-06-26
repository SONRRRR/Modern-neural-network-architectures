import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from .gpt_model import GPTModel
from clearml import Task

class GPTLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GPTModel(config)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.get('pad_token_id', 0))
        self.save_hyperparameters(config)
        self.gradient_clip_val = config.get('gradient_clip_val', 1.0)
    
    def forward(self, input_ids, segment_ids=None):
        return self.model(input_ids, segment_ids)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        segment_ids = batch.get('segment_ids', None)
        
        logits = self(input_ids, segment_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, self.config['vocab_size']),
            shift_labels.view(-1)
        )
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_perplexity', torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        segment_ids = batch.get('segment_ids', None)
        
        logits = self(input_ids, segment_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, self.config['vocab_size']),
            shift_labels.view(-1)
        )
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_perplexity', torch.exp(loss), on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # 1. Оптимизатор
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 3e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # 2. Scheduler с warm-up (всего 3 строки!)
        warmup_steps = self.config.get('warmup_steps', 1000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / warmup_steps)  # 1 строка!
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_val
        )

    def on_after_backward(self):
        """Логирование глобальной и локальных норм градиентов"""
        
        total_norm_squared = torch.zeros((), device=self.device)
        
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            
            grad_norm = parameter.grad.detach().norm(2)
            total_norm_squared += grad_norm.pow(2)
            
            # Логирование локальной нормы каждые 10 шагов
            if self.global_step % 10 == 0:
                safe_name = name.replace('.', '/')
                self.log(f'grad_norm/{safe_name}', grad_norm, on_step=True, on_epoch=False)
        
        # Логирование глобальной нормы
        global_norm = total_norm_squared.sqrt()
        self.log('grad_norm/global', global_norm, on_step=True, on_epoch=False)

    def setup_loggers_and_callbacks(self):
        """Настройка TensorBoard и ModelCheckpoint"""
        
        # TensorBoard логгер
        tensorboard_logger = TensorBoardLogger(
            save_dir=self.config.get('logs_dir', 'logs/'),
            name='gpt_model',
            version=None  # auto-increment
        )
        
        # ModelCheckpoint — сохраняет лучшую модель
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.get('checkpoint_dir', 'checkpoints/'),
            filename='gpt-{epoch:02d}-{val_perplexity:.2f}',
            monitor='val_perplexity',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        return tensorboard_logger, checkpoint_callback
    
    def setup_clearml(self):
        """Настройка ClearML для логирования эксперимента"""
        task = Task.init(
            project_name='GPT-Learning',
            task_name='GPTModel_Training',
            auto_connect_frameworks=True
        )
    
        # Сохранение конфига в ClearML
        task.connect(self.config)
    
        return task