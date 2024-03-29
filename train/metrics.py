import numpy as np

from dataset import Dataset


def multiclass_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Computes metrics for multiclass classification.

    Arguments:
        prediction, np array of int (num_samples) - model predictions.
        ground_truth, np array of int (num_samples) - true labels.

    Returns:
        accuracy - ratio of accurate predictions to total samples.
    """

    return np.round(np.sum(prediction == ground_truth) / len(prediction), decimals=3)


def compute_accuracy(model, dataset: Dataset) -> float:
    """Computes accuracy on provided data using mini-batches"""

    indices = np.arange(dataset.X.shape[0])
    sections = np.arange(dataset.batch_size, dataset.X.shape[0], dataset.batch_size)
    batches_indices = np.array_split(indices, sections)

    predict = np.zeros_like(dataset.y)

    for batch_indices in batches_indices:  # TODO rewrite for dataset like train
        model_pred = model.predict(dataset.X[batch_indices])
        predict[batch_indices] = model_pred.argmax(1)

    return multiclass_accuracy(predict, dataset.y)
