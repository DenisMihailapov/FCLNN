from __future__ import annotations

import numpy as np

from layers import FullyConnectedLayer
from utils.functions import softmax, softmax_with_cross_entropy, mse


class FCLayersNN:
    """ Neural network with fully connected layers """

    def __init__(self, n_input, n_output, n_hidden: int | list = None, reg=0):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        n_hidden, int - number of neurons in the hidden layer
        n_layers, int - number of layers in network
        learning_rate, float - learning rate
        reg, float - L2 regularization strength
        """

        self.reg = reg
        self.pred = None

        if n_hidden is None:
            n_hidden = []
        elif isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        elif not (isinstance(n_hidden, list) and all(isinstance(x, int) for x in n_hidden)):
            raise ValueError("Uncorrected value for n_hidden (need int or list of int)")

        self.n_layers = [n_input] + n_hidden + [n_output]
        self.fc_layers = [
            FullyConnectedLayer(self.n_layers[i - 1], self.n_layers[i], reg_strength=reg)
            for i in range(1, len(self.n_layers))
        ]

        self.fc_layers[-1].set_activation()

    def zero_grad(self):
        for fc_layer in self.fc_layers:
            fc_layer.zero_grad()

    def backward(self, d_pred):

        # [:0:-1]: [0, 1, 2, 3] -> [3, 2, 1]
        for fc_layer in self.fc_layers[:0:-1]:
            d_pred = fc_layer.backward(d_pred)

            # print("d_pred", d_pred[5])

        return self.fc_layers[0].backward(d_pred)

    def add_l2_regul_grad(self):
        for fc_layer in self.fc_layers:
            fc_layer.add_l2_regul_grad()

    def l2_regul_loss(self):
        s_loss = 0
        for fc_layer in self.fc_layers:
            s_loss += fc_layer.l2_regul_loss()

        return s_loss

    def compute_loss_and_gradients(self, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        if y.dtype in [np.int, np.uint8, np.bool]:
            loss, d_pred = softmax_with_cross_entropy(self.pred, y)
        else:
            y = y.reshape(self.pred.shape)
            loss, d_pred = mse(self.pred, y, get_gradient=True)

        self.backward(d_pred)

        if self.reg:
            self.add_l2_regul_grad()
            loss += self.l2_regul_loss()

        return np.round(loss, decimals=5)

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """

        self.pred = X
        for fc_layer in self.fc_layers[:-1]:
            self.pred = fc_layer(self.pred)

        self.pred = self.fc_layers[-1](self.pred)

        return softmax(self.pred).argmax(1)

    def params(self):
        params = dict()
        for i, fc_layer in enumerate(self.fc_layers):
            params[f'fc{i + 1}'] = fc_layer.params()

        return params
