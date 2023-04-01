import numpy as np

from nn.utils.functions import softmax_with_cross_entropy, mse


def compute_loss_and_gradients(pred, y):
    """
    Computes total loss and updates parameter gradients
    on a batch of training examples
    Arguments:
    X, np array (batch_size, input_features) - input data
    y, np array of int (batch_size) - classes
    """

    if y.dtype in [np.int, np.uint8, np.bool]:
        loss, d_pred = softmax_with_cross_entropy(pred, y)
    else:
        y = y.reshape(pred.shape)
        loss, d_pred = mse(pred, y, get_gradient=True)

    return loss, d_pred
