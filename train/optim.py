from abc import ABC
from typing import Tuple, Optional

import numpy as np

from nn.utils.train_param import Param
from .base_classes import Optimizer, LRScheduler


class SGD(Optimizer, ABC):
    """Implements SGD with momentum and weight decay"""

    def __init__(self, model_params: dict[str, dict[str, Param]], learning_rate: float = 0.1, weight_decay: float = 0.,
                 momentum: float = 0., lr_scheduler: Optional[LRScheduler] = None):

        self.learning_rate: float = learning_rate
        super().__init__(lr_scheduler)

        self.model_params = model_params

        self.weight_decay: float = weight_decay
        self.momentum: float = momentum

        self.velocity: dict[str, dict[str, np.ndarray]] = dict()
        for layer, param in model_params.items():
            self.velocity[layer] = {
                'weight': np.zeros_like(param["weight"].grad),
                'bias': np.zeros_like(param["bias"].grad)
            }

    def _update_rule(self, layer: str, param: Param, p_key: str) -> np.ndarray:

        p_grad = param.grad.copy()
        p_grad += self.weight_decay * param.value

        p_vel = self.velocity[layer]
        p_vel[p_key] = self.momentum * p_vel[p_key] + self.learning_rate * p_grad

        return param.value - p_vel[p_key]

    def zero_grad(self):
        for param in self.model_params.values():
            param["weight"].zero_grad()
            param["bias"].zero_grad()

    def step(self):
        for layer, param in self.model_params.items():
            param["weight"].value = self._update_rule(layer, param["weight"], "weight")
            param["bias"].value = self._update_rule(layer, param["bias"], "bias")

    def end_epoch(self):
        if self._lr_scheduler is not None:
            self._lr_scheduler.step_lr()


class AdamW(Optimizer, ABC):
    """Implements Adam with weight decay"""

    def __init__(self,
                 model_params: dict[str, dict[str, Param]],
                 learning_rate: float = 0.001,
                 lr_scheduler: Optional[LRScheduler] = None,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.01,
                 eps: float = 1e-8
                 ):
        self.learning_rate: float = learning_rate
        super().__init__(lr_scheduler)

        self.model_params = model_params

        self.weight_decay: float = weight_decay

        self.beta1 = betas[0]
        self.beta2 = betas[1]

        self.eps = eps

        self.time = 0

        self.moment: dict[str, dict[str, np.ndarray]] = dict()
        self.velocity: dict[str, dict[str, np.ndarray]] = dict()

        for layer, param in model_params.items():
            self.moment[layer] = {
                'weight': np.zeros_like(param["weight"].grad),
                'bias': np.zeros_like(param["bias"].grad)
            }

            self.velocity[layer] = {
                'weight': np.zeros_like(param["weight"].grad),
                'bias': np.zeros_like(param["bias"].grad)
            }

    def _update_rule(self, layer: str, param: Param, p_key: str) -> np.ndarray:

        self.moment[layer][p_key] = self.beta1 * self.moment[layer][p_key] + (1. - self.beta1) * param.grad
        self.velocity[layer][p_key] = self.beta2 * self.velocity[layer][p_key] + (1. - self.beta2) * param.grad ** 2

        moment = self.moment[layer][p_key] / (1. - self.beta1 ** self.time)
        velocity = self.velocity[layer][p_key] / (1. - self.beta2 ** self.time)

        p_grad = moment / (np.sqrt(velocity) + self.eps)
        return param.value - self.learning_rate * (p_grad + self.weight_decay * param.value)

    def zero_grad(self):
        for param in self.model_params.values():
            param["weight"].zero_grad()
            param["bias"].zero_grad()

    def step(self):
        for layer, param in self.model_params.items():
            param["weight"].value = self._update_rule(layer, param["weight"], "weight")
            param["bias"].value = self._update_rule(layer, param["bias"], "bias")

    def end_epoch(self):
        self.time += 1  # TODO: when update time?
        if self._lr_scheduler is not None:
            self._lr_scheduler.step_lr()
