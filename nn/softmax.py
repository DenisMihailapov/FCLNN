from abc import ABC

import numpy as np

from nn import IFunc
from nn.utils.functions import softmax


class Softmax(IFunc, ABC):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        self.probs = softmax(x)
        return self.probs

    def backward(self, d_out):
        """Calculate the jacobian of the Softmax function for the given set of inputs."""
        sm_d_out = d_out.copy()
        eye = np.eye(d_out.shape[1])
        for prob, d_out_, batch in zip(self.probs, d_out, range(d_out.shape[0])):
            grad = prob[:, np.newaxis] * (eye - prob[np.newaxis, :])
            sm_d_out[batch] = grad.dot(d_out_)

        return sm_d_out
