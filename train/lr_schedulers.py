from abc import ABC


class LRScheduler(ABC):
    def __init__(self, init_lr):
        """Abstract base class for all Scheduler objects."""
        self.cur_lr = None
        self.step = None
        self.init_lr = init_lr
        self.reset_schedule()

    def __call__(self):
        self.step_lr()
        return self.cur_lr

    def step_lr(self):
        raise NotImplementedError

    def reset_schedule(self):
        self.step = 0
        self.cur_lr = self.init_lr


class ConstantScheduler(LRScheduler):

    def __str__(self):
        return f"ConstantScheduler(lr={self.cur_lr})"

    def step_lr(self):
        self.step += 1
        return self.cur_lr


class LinearScheduler(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, init_lr, last_lr, num_epochs):
        super().__init__(init_lr)
        self.last_lr = last_lr
        self.step_val = (last_lr - init_lr) / num_epochs

        self.num_epochs = num_epochs

    def __str__(self):
        return f"LinearScheduler(init_lr={self.init_lr}, last_lr={self.last_lr}, num_epochs={self.num_epochs})"

    def step_lr(self):
        if self.step >= self.num_epochs:
            raise StopIteration

        self.cur_lr += self.step_val
        self.step += 1
        return self.cur_lr
