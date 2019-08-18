# Author: borgwang <borgwang@126.com>
#
# Filename: run.py
# Description: Comparison between tinynn and Tensorflow on MNIST.

import runtime_path  # isort:skip

import gzip
import os
import pickle
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

from core.evaluator import AccEvaluator
from core.layers import Dense
from core.layers import ReLU
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    path = os.path.join(DATA_DIR, url.split("/")[-1])

    # download
    try:
        if os.path.exists(path):
            print("{} already exists.".format(path))
        else:
            print("Downloading {}.".format(url))
            try:
                urlretrieve(url, path)
            except URLError:
                raise RuntimeError("Error downloading resource!")
            finally:
                print()
    except KeyboardInterrupt:
        print("Interrupted")

    # load
    print("Loading MNIST dataset.")
    with gzip.open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


BATCH_SIZE = 128
LR = 1e-3
NUM_EPS = 20
DATA_DIR = "./examples/mnist/data"


def run_tinynn(dataset):
    train_x, train_y, test_x, test_y = dataset

    net = Net([
        Dense(784, 400),
        ReLU(),
        Dense(400, 100),
        ReLU(),
        Dense(100, 10),
    ])
    model = Model(net=net,
                  loss=SoftmaxCrossEntropyLoss(),
                  optimizer=Adam(lr=LR))

    iterator = BatchIterator(batch_size=BATCH_SIZE)
    evaluator = AccEvaluator()
    for epoch in range(NUM_EPS):
        for batch in iterator(train_x, train_y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grad(grads)
        # evaluate
        test_pred = model.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print("Epoch %d \t %s" % (epoch, res))


def run_tf(dataset):
    train_x, train_y, test_x, test_y = dataset

    sess = tf.Session()
    x_ph, y_ph, is_train_ph, train_op, out = tf_dense()
    sess.run(tf.global_variables_initializer())

    iterator = BatchIterator(batch_size=BATCH_SIZE)
    evaluator = AccEvaluator()
    for epoch in range(NUM_EPS):
        for batch in iterator(train_x, train_y):
            sess.run(train_op, feed_dict={
                x_ph: batch.inputs, y_ph: batch.targets, is_train_ph: True})
        # evaluate
        test_pred = sess.run(out, feed_dict={x_ph: test_x, is_train_ph: False})
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print("Epoch %d \t %s" % (epoch, res))


def tf_dense():
    with tf.variable_scope("dense"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
        is_train_ph = tf.placeholder(dtype=bool)
        fc1 = tf.layers.dense(x, 400, activation=tf.nn.relu)
        fc1 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)
        out = tf.layers.dense(fc1, 10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=out, labels=y))
        train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    return x, y, is_train_ph, train_op, out


def main():
    train_set, valid_set, test_set = prepare_dataset()
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    dataset = train_x, train_y, test_x, test_y
    print("Running Tensorflow")
    run_tf(dataset)
    print("Running tinynn")
    run_tinynn(dataset)


if __name__ == "__main__":
    main()
