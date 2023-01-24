

class IdentityFunc:

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, d_out):
        return d_out

    def params(self):
        return {}


class ReLUFunction(IdentityFunc):
    def __init__(self):
        self.X_mask = None

    def forward(self, x):
        self.X_mask = (x > 0.0)
        return self.X_mask * x

    def backward(self, d_out):
        return self.X_mask * d_out
