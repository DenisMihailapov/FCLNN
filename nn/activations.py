from abc import ABC

import numpy as np

from nn.utils.functions import sigmoid, softmax


class IFunc:

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_out):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError


class Identity(IFunc, ABC):

    def forward(self, x):
        return x

    def backward(self, d_out):
        return d_out

    def params(self):
        return {}


class ReLU(IFunc, ABC):
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0.0)
        return self._mask * x

    def backward(self, d_out):
        grad = self._mask
        return grad * d_out


class LeakyReLU(IFunc, ABC):
    def __init__(self, negative_slope=0.01):
        self._mask = None
        self.negative_slope = negative_slope

    def forward(self, x):
        self._mask = np.float32(x >= 0.0) + self.negative_slope * np.float32(x < 0.0)
        return self._mask * x

    def backward(self, d_out):
        grad = self._mask
        return grad * d_out


class Sigmoid(IFunc, ABC):

    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = sigmoid(x)
        return self._out

    def backward(self, d_out):
        grad = self._out * (1. - self._out)
        return grad * d_out


class SiLU(IFunc, ABC):

    def __init__(self):
        self._x = None
        self._sigm = None
        self._out = None

    def forward(self, x):
        self._x = x
        self._sigm = sigmoid(x)
        self._out = x * self._sigm
        return self._out

    def backward(self, d_out):
        grad = self._sigm * (1. + self._x - self._out)
        return grad * d_out


class Softmax(IFunc, ABC):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        self.probs = softmax(x)
        return self.probs

    def backward(self, d_out):
        grad = self.forward(self.probs) * (1 - self.forward(self.probs))
        return grad * d_out
