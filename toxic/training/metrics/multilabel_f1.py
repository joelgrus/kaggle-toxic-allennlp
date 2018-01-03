from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("multilabel-f1")
class MultiLabelF1Measure(Metric):
    """
    Computes multilabel F1. Assumes that predictions are 0 or 1.
    """
    def __init__(self) -> None:
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.LongTensor,
                 gold_labels: torch.LongTensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        gold_labels : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        """
        self._true_positives += (predictions * gold_labels).sum()
        self._false_positives += (predictions * (1 - gold_labels)).sum()
        self._true_negatives += ((1 - predictions) * (1 - gold_labels)).sum()
        self._false_negatives += ((1 - predictions) * gold_labels).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
