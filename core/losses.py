# Author: borgwang <borgwang@126.com>
#
# Filename: losses.py
# Description: Implementation of common loss functions in neural network.


import numpy as np


class BaseLoss(object):

    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSELoss(BaseLoss):

    def loss(self, predicted, actual):
        m = predicted.shape[0]
        return 0.5 * np.sum((predicted - actual) ** 2) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return (predicted - actual) / m


class SoftmaxCrossEntropyLoss(BaseLoss):
    """L = (-log(exp(x[class]) / sum(exp(x))))"""

    def loss(self, predicted, actual):
        m = predicted.shape[0]
        # Softmax
        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)
        # cross entropy loss
        nll = -np.log(np.sum(p * actual, axis=1))

        return np.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = np.copy(predicted)
        grad[range(m), actual] -= 1.0
        return grad / m
