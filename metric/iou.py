__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2024

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import torch
import numpy as np
from metric import metric
from metric.confusionmatrix import ConfusionMatrix


class IoU(metric.Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)
        self.counter = 0
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.reshape(-1), target.reshape(-1))
        self.counter += 1

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                # conf_matrix[:, index] = 0  # pred label, shouldn't be zeroed
                conf_matrix[index, :] = 0  # gt label
        self.true_positive = np.diag(conf_matrix)
        # row: gt, column: pred,
        # FP: column sum and should ignore unlabeled row,
        # FN: row sum and should include unlabeled column
        self.false_positive = np.sum(conf_matrix, 0) - self.true_positive
        self.false_negative = np.sum(conf_matrix, 1) - self.true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = self.true_positive / (self.true_positive + self.false_positive + self.false_negative)

        # mIoU should be computed over valid classes only!
        miou = []
        for i, item in enumerate(iou):
            if i in self.ignore_index:
                continue
            miou.append(item)
        miou = np.nanmean(np.array(miou))

        return iou, miou


class IoU3D(IoU):
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__(num_classes, normalized=normalized, ignore_index=ignore_index)

    def add(self, predicted, target):
        """Adds the predicted and target pair to the 3D IoU metric.

        Keyword arguments:
        - predicted (Tensor):  (V,) tensor of integer values between 0 and K-1 or (V, K) tensor.
        - target (Tensor): (V,) tensor of integer values between 0 and K-1 or (V, K) tensor.

        """
        # Dimensions check
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 1 or predicted.dim() == 2, \
            "predictions must be of dimension (V,) or (V, K)"
        assert target.dim() == 1 or predicted.dim() == 2, \
            "targets must be of dimension (V,) or (V, K)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 2:
            _, predicted = predicted.max(1)  # [V,]
        if target.dim() == 2:
            _, target = target.max(1)  # [V,]

        self.conf_metric.add(predicted, target)
        # sanity check
        # self.conf_metric.add_naive(predicted, target)
