from abc import ABC

import numpy as np

from nn.utils.train_param import Param


class Optimizer:
    """Implements differed optimization rules for weights update."""

    def _update_rule(self, layer: str, param: Param, p_key: str) -> np.ndarray:
        """weights update rule"""
        raise NotImplementedError

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        raise NotImplementedError

    def step(self):
        """Update the parameter values using the update rule."""
        raise NotImplementedError


class SGD(Optimizer, ABC):
    """Implements SGD with momentum and weight decay"""

    def __init__(self,
                 model_params: dict[str, dict[str, Param]],
                 learning_rate: float, weight_decay: float = 0., momentum: float = 0.
                 ):

        self.model_params = model_params

        self.learning_rate: float = learning_rate
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
