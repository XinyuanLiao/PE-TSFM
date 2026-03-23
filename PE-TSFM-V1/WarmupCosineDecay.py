from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineDecay(_LRScheduler):
    def __init__(self, optimizer, init_value, peak_value, warmup_steps, decay_steps, end_value=1e-7, exponent=1.0, last_epoch=-1):
        self.init_value = init_value
        self.peak_value = peak_value
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_value = end_value
        self.exponent = exponent
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [self.init_value + (self.peak_value - self.init_value) * self.last_epoch / self.warmup_steps]
        else:
            # Cosine decay with exponent
            decay_epoch = self.last_epoch - self.warmup_steps
            progress = min(decay_epoch / self.decay_steps, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decay_factor = (self.peak_value - self.end_value) * cosine_decay ** self.exponent + self.end_value
            return [decay_factor]