import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    exp = np.exp(predictions - np.max(predictions))

    # probs for (N) shape predictions
    if predictions.shape == (len(predictions),):
        probs = exp / np.sum(exp)

    # probs for (batch_size, N) predictions
    else:
        probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array  - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def mse(x, y, get_gradient=False):
    """Функция измерения среднеквадратичной ошибки"""
    d = x - y
    if get_gradient:
        return np.mean(d ** 2), 2 * d
    return np.mean(d ** 2)
