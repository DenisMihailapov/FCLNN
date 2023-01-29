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


def cross_entropy_loss(probs, target_index):
    """Computes cross-entropy loss.

    Arguments:
        probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class.

        target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s).

    Returns:
        loss: single value.
    """

    # loss for (N) shape probs
    if probs.shape == (len(probs),):
        loss = - np.log(probs[target_index])

    # loss for (batch_size, N) shape probs
    else:
        loss = - np.log(probs[np.arange(len(probs)), target_index])
        # use matrix coordinates where
        #    colums from target_index,
        #    rows   from np.arange(len(probs))

    return loss


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


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions: np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index).mean()
    mask = np.zeros_like(predictions)

    # mask and dprediction for (N) shape predictions
    if predictions.shape == (len(predictions),):
        mask[target_index] = 1
        d_preds = - (mask - probs)

    # mask and dprediction for (batch_size, N) shape predictions
    else:
        mask[np.arange(len(mask)), target_index] = 1
        d_preds = - (mask - probs) / len(mask)

    return loss, d_preds


def mse(x, y, get_gradient=False):
    """Функция измерения среднеквадратичной ошибки"""
    d = x - y
    if get_gradient:
        return np.mean(d ** 2), 2 * d
    return np.mean(d ** 2)
