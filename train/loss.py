from abc import ABC

import numpy as np

from nn.utils.functions import softmax
from train.base_classes import Loss


class CrossEntropyLoss(Loss):

    def __init__(self):
        pass

    def _loss_fn(self, probs: np.ndarray, target_index: np.ndarray):
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
            probs = probs[target_index]
        # loss for (batch_size, N) shape probs
        else:
            probs = probs[np.arange(len(probs)), target_index]

        return - np.log(probs + np.finfo(float).eps).mean()

    def _backward(self, probs: np.ndarray, target_index: np.ndarray):
        """Computes cross-entropy backpropagation.

        Arguments:
            probs, np array, shape is either (N) or (batch_size, N) -
            probabilities for every class.

            target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s).

        Returns:
            .
        """

        mask = np.zeros_like(probs)

        # mask and dprediction for (N) shape predictions
        if probs.shape == (len(probs),):
            mask[target_index] = 1
            batch_size = 1

        # mask and dprediction for (batch_size, N) shape predictions
        else:
            mask[np.arange(len(mask)), target_index] = 1
            batch_size = len(mask)

        return (probs - mask) / batch_size

    def forward(self, predictions: np.ndarray, target_index: np.ndarray):
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

        loss = self._loss_fn(probs, target_index)
        d_preds = self._backward(probs, target_index)

        return loss, d_preds
