from typing import Tuple

import numpy as np

from nn.utils.functions import he_normal, truncated_normal, he_uniform
from nn.utils.train_param import Param


class ParamsInit:

    def __init__(self,
                 mode: str = 'normal',
                 mean: float = 0., std: float = 0.1,
                 low: float = -0.4, high: float = 0.4
                 ):
        self.mode = mode
        self.mean = mean
        self.std = std

        self.low = low
        self.high = high

    def from_normal(self, shape: Tuple) -> Param:

        if self.mode == "normal":
            return Param(self.mean + self.std * np.random.randn(*shape))
        if self.mode == "truncated_normal":
            return Param(truncated_normal(self.mean, self.std, shape))
        elif self.mode == "he_normal":
            return Param(he_normal(shape))
        else:
            raise ValueError("Unrecognized initialization mode: {}".format(self.mode))

    def from_uniform(self, shape: Tuple) -> Param:

        if self.mode == "uniform":
            return Param(np.random.uniform(self.low, self.high, size=shape))
        elif self.mode == "he_uniform":
            return Param(he_uniform(shape))
        else:
            raise ValueError("Unrecognized initialization mode: {}".format(self.mode))

    def __call__(self, shape: Tuple) -> Param:

        if "normal" in self.mode:
            return self.from_normal(shape)
        if "uniform" in self.mode:
            return self.from_uniform(shape)
        else:
            raise ValueError("Unrecognized initialization mode: {}".format(self.mode))
