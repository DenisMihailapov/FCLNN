from __future__ import annotations

from abc import ABC

import numpy as np

from nn.activations import IFunc, get_activation
from nn.utils.initializer import ParamsInit


class FullyConnectedLayer(IFunc, ABC):
    def __init__(self, n_input, n_output, reg_strength, activation="sigmoid"):
        self.reg = reg_strength
        self.param_init = ParamsInit(mode="he_uniform")

        self.weight = self.param_init((n_input, n_output))
        self.bias = self.param_init((1, n_output))
        self.x = None

        self.act = get_activation(activation)

    def reset_activation(self, activation="ident"):
        self.act = get_activation(activation)

    def forward(self, x):
        self.x = x
        return self.act(self.bias + self.weight.dot(x))

    def backward(self, d_out):
        """Computes gradient with respect to input and accumulates gradients within self.weight and self.b

        Arguments:
            d_out, np array (batch_size, n_output) - gradient of loss function with respect to output.

        Returns:
            d_result: np array (batch_size, n_input) - gradient with respect to input.
        """

        d_out = self.act.backward(d_out)  # d_act

        self.weight.grad = np.dot(self.x.T, d_out)
        self.bias.grad = np.dot(np.ones((1, self.x.shape[0])), d_out)

        d_out = np.dot(d_out, self.weight.T())  # d_X
        return d_out

    def params(self):
        return {'weight': self.weight, 'bias': self.bias}

    def l2_regul_loss(self):
        if self.reg is None:
            self.reg = 0
        if self.reg == 0:
            return np.zeros(1)

        return self.reg * (np.sum(self.weight ** 2) + np.sum(self.bias ** 2))

    def add_l2_regul_grad(self):
        if self.reg is None:
            self.reg = 0
        if self.reg == 0:
            return

        self.weight.grad += self.weight * (2 * self.reg)
        self.bias.grad += self.bias * (2 * self.reg)

    def zero_grad(self):
        self.weight.zero_grad()
        self.bias.zero_grad()

    def add_grad(self, W_grad, b_grad):
        self.weight.grad += W_grad
        self.bias.grad += b_grad
