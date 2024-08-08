import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineLR(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupCosineLR, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1, self.warmup_epochs))
        else:
            # Cosine annealing
            progress = float(epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))