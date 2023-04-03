from abc import ABC

from nn import IFunc
from nn.utils.functions import softmax


class Softmax(IFunc, ABC):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        self.probs = softmax(x)
        return self.probs

    def backward(self, d_out):
        grad = self.forward(self.probs) * (1 - self.forward(self.probs))
        return grad * d_out
