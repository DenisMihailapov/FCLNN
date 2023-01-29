from __future__ import annotations

import numpy as np


class Param:
    """Trainable parameter of the model.
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value: np.ndarray = value
        self.shape = value.shape
        self.grad: np.ndarray = np.zeros(self.shape)

    @staticmethod
    def _get_val(other):
        return other.value if isinstance(other, Param) else other

    def __mul__(self, other: int | float | Param | np.ndarray):
        return self.value * self._get_val(other)

    def dot(self, other: np.ndarray | Param):
        return np.dot(self._get_val(other), self.value)

    def __add__(self, other: np.ndarray | Param):
        return self.value + self._get_val(other)

    def __pow__(self, degree: int | float ):
        return self.value ** degree

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def T(self):
        return self.value.T
