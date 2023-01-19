from typing import Callable


class Optimizer:
    """
    Implements differed optimization rules for weights update
    """

    def __init__(self, model_params, learning_rate, opt_type='sgd'):

        assert opt_type == 'sgd'
        self.opt_type: str = opt_type
        self.model_params: dict = model_params
        self.learning_rate: float = learning_rate

    def _sgd(self, w, d_w) -> float:
        """vanilla SGD"""
        return w - d_w * self.learning_rate

    def get_opt_func(self) -> Callable[[float, float], float]:

        if self.opt_type == 'sgd':
            return self._sgd
        else:
            raise NotImplementedError(f"optimization rule {self.opt_type} is not Implemented")

    def zero_grad(self):
        for param in self.model_params.values():
            param["W"].zero_grad()
            param["B"].zero_grad()

    def step(self):

        opt_func = self.get_opt_func()
        for param in self.model_params.values():
            # print("W.grad", param["W"].grad[:5])
            # print("W.v", param["W"].value, "\nW.g", param["W"].grad, "\nlr", self.learning_rate)
            param["W"].value = opt_func(param["W"].value, param["W"].grad)
            param["B"].value = opt_func(param["B"].value, param["B"].grad)
