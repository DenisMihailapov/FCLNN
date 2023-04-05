from __future__ import annotations

from abc import ABC
from typing import Optional

from numpy import ndarray

from nn.utils.train_param import Param


class Optimizer(ABC):
    pass


class LRScheduler(ABC):
    def __init__(self, init_lr):
        """Abstract base class for all Scheduler objects."""

        self._optimizer: Optional[Optimizer] = None
        self.init_lr = init_lr
        self.cur_lr = init_lr
        self.step = 0

    def __call__(self) -> float:
        self.step_lr()
        return self.cur_lr

    def _update_lr(self):
        """Update rule for cur_lr."""
        raise NotImplementedError

    def reset_schedule(self):
        self.step = 0
        self.cur_lr = self.init_lr

    def set_optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer

    def step_lr(self) -> float:
        self._update_lr()
        self.step += 1
        if self._optimizer is not None:  # synchronize lr with optimizer
            self._optimizer.set_lr(self.cur_lr)
        return self.cur_lr


# define Optimizer
class Optimizer(ABC):
    """Implements differed optimization rules for weights update."""
    learning_rate: float

    def __init__(self, lr_scheduler: Optional[LRScheduler] = None):
        self._lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self._lr_scheduler.set_optimizer(self)
            self.set_lr(lr_scheduler.cur_lr)

    def set_lr(self, learning_rate):
        self.learning_rate = learning_rate

    def _update_rule(self, layer: str, param: Param, p_key: str) -> ndarray:
        """weights update rule"""
        raise NotImplementedError

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        raise NotImplementedError

    def step(self):
        """Update the parameter values using the update rule."""
        raise NotImplementedError

    def end_epoch(self):
        """Rule for end epoch."""
        raise NotImplementedError
