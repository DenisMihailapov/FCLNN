import numpy as np

from nn.utils.functions import he_normal, truncated_normal
from nn.utils.train_param import Param


class ParamsInit:

    def __init__(self, mode='normal'):
        self.mode = mode

    def __call__(self, shape: tuple):

        if self.mode == "normal":
            return Param(0.1 * np.random.randn(*shape))
        if self.mode == "truncated_normal":
            return Param(truncated_normal(0, 0.1, shape))
        elif self.mode == "he_normal":
            return Param(he_normal(shape))
        else:
            raise ValueError("Unrecognized initialization mode: {}".format(self.mode))
