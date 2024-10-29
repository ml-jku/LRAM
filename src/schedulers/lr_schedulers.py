from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLRSingleCycle(CosineAnnealingLR):

    def get_lr(self):
        # in case T_max is reached, always return eta_min, don't go up in the cycle again
        lrs = super().get_lr()
        if self.last_epoch >= self.T_max:
            lrs = [self.eta_min for _ in self.optimizer.param_groups]
        return lrs


def make_lr_scheduler(optimizer, kind="cosine", sched_kwargs=None):
    if sched_kwargs is None:
        sched_kwargs = {}
    if kind == "cosine":
        return CosineAnnealingLRSingleCycle(optimizer, **sched_kwargs)
    elif kind == "cosine_restart":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        return CosineAnnealingWarmRestarts(optimizer, **sched_kwargs)
    elif kind == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, **sched_kwargs)
    elif kind == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, **sched_kwargs)
    elif kind == "cyclic":
        from torch.optim.lr_scheduler import CyclicLR
        return CyclicLR(optimizer, cycle_momentum=False, **sched_kwargs)
    elif kind == "exp":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, **sched_kwargs)
    raise ValueError(f"Unknown scheduler {kind}")
