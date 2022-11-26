from __future__ import annotations

import numpy as np


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.shape = value.shape
        self.grad = np.zeros(self.shape)

    def _get_val(self, other):
        return other.value if isinstance(other, Param) else other

    def __mul__(self, other: int | float | Param):
        return self.value * self._get_val(other)

    def dot(self, other):
        return np.dot(self._get_val(other), self.value)

    def __add__(self, other):
        return self.value + self._get_val(other)

    def __pow__(self, degree):
        return self.value ** degree

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def T(self):
        return self.value.T

class IFunction:

    def forward(self, X):
        return X

    def __call__(self, X):
        return self.forward(X)

    def backward(self, d_out):
        return d_out

    def params(self):
        return {}

class ReLUFunction(IFunction):
    def __init__(self):
        self.X_mask = None

    def forward(self, X):
        self.X_mask = (X > 0.0)
        return self.X_mask * X

    def backward(self, d_out):
        return self.X_mask * d_out


class FullyConnectedLayer:
    def __init__(self, n_input, n_output, reg_strength, activation="relu"):
        self.reg = reg_strength

        self.W = Param(0.1 * np.random.randn(n_input, n_output))
        self.B = Param(0.1 * np.random.randn(1, n_output))
        self.X = None

        self.set_activation(activation)
    
    
    def set_activation(self, activation=None):
        if activation is None:
            self.act = IFunction()
        elif activation=="relu":
            self.act = ReLUFunction()     
        else:
            raise NotImplementedError(f"Unknown activation {activation}")  

    def forward(self, X):
        self.X = X
        return self.act(self.B + self.W.dot(X))

    def __call__(self, X):
        return self.forward(X)

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
          of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        #print("d_out", d_out[:5, :5])
        d_out = self.act.backward(d_out) # d_act

        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1, self.X.shape[0])), d_out)

        # print("self.W.grad", self.W.grad)
        # print("self.B.grad", self.W.grad)

        d_out = np.dot(d_out, self.W.T()) # d_X
        return d_out

    def params(self):
        return {'W': self.W, 'B': self.B}

    def l2_regul_loss(self):
        if self.reg is None:
            self.reg = 0
        if self.reg == 0:
            return np.zeros(1)

        return self.reg * (np.sum(self.W ** 2) + np.sum(self.B ** 2))

    def add_l2_regul_grad(self):
        if self.reg is None:
            self.reg = 0
        if self.reg == 0:
            return

        self.W.grad += self.W * (2 * self.reg)
        self.B.grad += self.B * (2 * self.reg)

    def zero_grad(self):
        self.W.zero_grad()
        self.B.zero_grad()

    def add_grad(self, W_grad, B_grad):
        self.W.grad += W_grad
        self.B.grad += B_grad
