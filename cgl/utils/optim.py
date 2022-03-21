import numpy as np
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, peak_lr=None, end_lr=None):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if self.end_lr is not None:
            if epoch < self.max_num_iters:
                lr = (self.peak_lr - self.end_lr) * 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters)) + self.end_lr
                lr_factor = lr / self.peak_lr
            else:
                lr_factor = self.end_lr / self.peak_lr
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if self.warmup != 0:
            if epoch <= self.warmup:
                lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor



class PolynomialDecayLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False