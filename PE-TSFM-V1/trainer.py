import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        lr_scheduler=None,  
        max_epochs=100, 
        use_early_stopping=True, 
        use_amp=True,
        early_stopping_patience=10,
        device="cuda",
        scheduler_step_per_batch=True,
        model_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.best_val_loss = float('inf')
        self.scheduler_step_per_batch = scheduler_step_per_batch
        
        if model_name is not None:
            self.writer = SummaryWriter(log_dir='./logs/' + model_name)

    def pretrain(self, train_loader):
        self.model.to(self.device)
        self.patience_counter = 0
        global_step = 0

        with tqdm(desc="Pre-training", total=self.max_epochs) as pbar:
            for epoch in range(1, self.max_epochs + 1):
                self.model.train()
                total_loss = 0.0

                for (inputs,) in train_loader:
                    x = inputs.to(self.device)

                    self.optimizer.zero_grad()

                    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                        loss = self.model(x.permute(0, 2, 1)).loss
                    
                    self.writer.add_scalar("Loss/pretrain", loss.item(), global_step)
                    global_step += 1

                    total_loss += loss.item()

                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    if self.lr_scheduler and self.scheduler_step_per_batch:
                        self.lr_scheduler.step()

                avg_train_loss = total_loss / len(train_loader)

                if self.lr_scheduler and not self.scheduler_step_per_batch:
                    self.lr_scheduler.step()

                pbar.update()
                pbar.set_postfix_str(
                    f"Train Loss: {avg_train_loss:.4f} "
                )

    def finetune(self, train_loader, val_loader=None):
        self.model.to(self.device)
        self.patience_counter = 0

        with tqdm(desc="Fine-tuning", total=self.max_epochs) as pbar:
            for epoch in range(1, self.max_epochs + 1):
                self.model.train()
                total_loss = 0.0
                total_correct = 0
                for inputs, targets in train_loader:
                    x, y = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()

                    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                        outputs = self.model(x.permute(0, 2, 1), labels=y)

                    loss = outputs.loss
                    logits = outputs.prediction_logits
                    preds = torch.argmax(logits, dim=-1)
                    total_correct += (preds == y).sum().item()
                    total_loss += loss.item()

                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    if self.lr_scheduler and self.scheduler_step_per_batch:
                        self.lr_scheduler.step()

                avg_train_loss = total_loss / len(train_loader)

                if self.lr_scheduler and not self.scheduler_step_per_batch:
                    self.lr_scheduler.step()

                if val_loader:
                    val_loss, val_acc = self.evaluate(val_loader)
                else:
                    val_loss = None

                pbar.update()
                pbar.set_postfix_str(
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {100 * total_correct / len(train_loader.dataset):.2f}%, "
                    f"{'Val Loss: ' + str(round(val_loss, 4)) if val_loss is not None else ''}, "
                    f"{'Val Acc: ' + str(round(val_acc, 2)) if val_acc is not None else 'N/A'}%"
                )

                # Early Stopping
                if self.use_early_stopping and val_loader:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        best_model_state = self.model.state_dict()
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print("Early stopping triggered.")
                            self.model.load_state_dict(best_model_state)
                            break

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x.permute(0, 2, 1), labels=y)
                loss = output.loss
                logits = output.prediction_logits
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == y).sum().item()
                total_loss += loss.item()
        return total_loss / len(val_loader), total_correct / len(val_loader.dataset) * 100
