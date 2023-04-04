from typing import Tuple

import numpy as np


def softmax(predictions: np.ndarray) -> np.ndarray:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))


def l2_regularization(W: np.ndarray, reg_strength: float) -> Tuple[float, np.ndarray]:
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


def calc_in_out_shape(weight_shape: Tuple):
    """
    Compute number of input and output for a weight matrix/volume.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.
    Returns
    -------
    n_input : int
        The number of input units in the weight tensor
    n_output : int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        n_input, n_output = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        n_input, n_output = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    return n_input, n_output


def truncated_normal(mean: float, std: float, out_shape: Tuple):
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples


# ----------------------------------------------- #
#              Weight Initialization              #
# ----------------------------------------------- #

def he_uniform(weight_shape: Tuple):
    n_input, _ = calc_in_out_shape(weight_shape)
    b = np.sqrt(6 / n_input)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape: Tuple):
    n_input, _ = calc_in_out_shape(weight_shape)
    std = np.sqrt(2 / n_input)
    return truncated_normal(0, std, weight_shape)
