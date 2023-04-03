import numpy as np
from tqdm import tqdm

from dataset import Dataset
from model import FCLayersNN
from .loss import Loss
from .metrics import compute_accuracy
from .optim import Optimizer


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch Optimizer with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model: FCLayersNN, dataset: Dataset,
                 optim: Optimizer, loss_fn: Loss,
                 log_freq=5, num_epochs=20
                 ):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.model: FCLayersNN = model
        self.dataset: Dataset = dataset
        self.num_epochs = num_epochs
        self.log_freq = log_freq

        self.optimizer: Optimizer = optim  # TODO lr change strategy
        # self.learning_rate = learning_rate
        # self.learning_rate_decay = learning_rate_decay
        self.loss_fn = loss_fn

    def epoch_step(self):
        batch_losses = []
        self.dataset.mode("train")

        for X, y in self.dataset:
            self.optimizer.zero_grad()

            logits = self.model.predict(X)
            loss, d_pred = self.loss_fn(logits, y)

            self.model.backward(d_pred)
            loss += self.model.l2_regularization()

            batch_losses.append(np.round(loss, decimals=5))

            self.optimizer.step()

        ave_loss = np.mean(batch_losses)

        train_accuracy = compute_accuracy(self.model, self.dataset)

        return batch_losses[-1], ave_loss, train_accuracy

    def fit(self):
        """
        Trains a model
        """

        loss_history, train_acc_history, val_acc_history = [], [], []

        for epoch in tqdm(range(self.num_epochs)):

            self.dataset.mode("train")
            last_batch_loss, ave_loss, train_accuracy = self.epoch_step()

            self.dataset.mode("val")
            val_accuracy = compute_accuracy(self.model, self.dataset)

            if epoch % self.log_freq == 0:
                if self.dataset.y.dtype in [np.int, np.uint8, np.bool]:
                    print(f"\n Loss: {last_batch_loss}, Train accuracy: {train_accuracy}, val accuracy: {val_accuracy}")
                else:
                    print(f"\n Loss: {last_batch_loss}")
                    train_accuracy, val_accuracy = -1, -1

            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history
