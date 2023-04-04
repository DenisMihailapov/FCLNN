from abc import ABC

import numpy as np

from nn.utils.functions import sigmoid


class IFunc:

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self):
        raise NotImplementedError


class Identity(IFunc, ABC):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out

    def params(self):
        return {}


class ReLU(IFunc, ABC):
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0.0)
        return self._mask * x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        grad = self._mask
        return grad * d_out


class LeakyReLU(IFunc, ABC):
    def __init__(self, negative_slope=0.01):
        self._mask = None
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = np.float32(x >= 0.0) + self.negative_slope * np.float32(x < 0.0)
        return self._mask * x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        grad = self._mask
        return grad * d_out


class Sigmoid(IFunc, ABC):

    def __init__(self):
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = sigmoid(x)
        return self._out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        grad = self._out * (1. - self._out)
        return grad * d_out


class SiLU(IFunc, ABC):

    def __init__(self):
        self._x = None
        self._sigm = None
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._sigm = sigmoid(x)
        self._out = x * self._sigm
        return self._out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        grad = self._sigm * (1. + self._x - self._out)
        return grad * d_out


def get_activation(activation: str = "ident") -> IFunc:
    if activation == "ident":
        return Identity()

    elif activation == "relu":
        return ReLU()
    elif activation == "leaky_relu":
        return LeakyReLU()
    elif activation == "p_relu":
        raise NotImplementedError("TODO")

    elif activation == "sigmoid":
        return Sigmoid()
    elif activation == "silu":
        return SiLU()

    else:
        raise NotImplementedError(f"Unknown activation {activation}")
