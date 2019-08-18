# Author: borgwang <borgwang@126.com>
#
# Filename: run.py
# Description: Comparison between tinynn and Tensorflow on MNIST.

import runtime_path  # isort:skip

import argparse
import gzip
import pickle
import time
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

from core.evaluator import AccEvaluator
from core.layers import Conv2D
from core.layers import Dense
from core.layers import Flatten
from core.layers import MaxPool2D
from core.layers import ReLU
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    path = os.path.join(data_dir, url.split("/")[-1])

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


def run_tinynn(model_type, dataset):
    train_x, train_y, test_x, test_y = dataset

    if model_type == "cnn":
        net = Net([
            Conv2D(kernel=[5, 5, 1, 16], stride=[2, 2], padding="VALID"),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2], padding="VALID"),
            Conv2D(kernel=[3, 3, 16, 32], padding="VALID"),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2], padding="VALID"),
            Flatten(),
            Dense(10)
        ])
    elif model_type == "dense":
        net = Net([
            Dense(200),
            ReLU(),
            Dense(100),
            ReLU(),
            Dense(70),
            ReLU(),
            Dense(30),
            ReLU(),
            Dense(10)
        ])
    else:
        raise ValueError("Invalid argument model_type! Must be 'cnn' or 'dense'")
    model = Model(net=net, loss=SoftmaxCrossEntropyLoss(), optimizer=Adam(lr=1e-4))

    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grad(grads)
        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase("TEST")
        test_pred = model.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print(res)
        print("-" * 10)


def run_tf(model_type, dataset):
    train_x, train_y, test_x, test_y = dataset

    if model_type == "cnn":
        sess = tf.Session()
        x_ph, y_ph, is_train_ph, train_op, out = tf_cnn()
        sess.run(tf.global_variables_initializer())

        iterator = BatchIterator(batch_size=args.batch_size)
        evaluator = AccEvaluator()
        for epoch in range(args.num_ep):
            t_start = time.time()
            for batch in iterator(train_x, train_y):
                sess.run(train_op, feed_dict={
                    x_ph: batch.inputs, y_ph: batch.targets, is_train_ph: True})
            print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
            # evaluate
            test_pred = sess.run(out, feed_dict={x_ph: test_x, is_train_ph: False})
            test_pred_idx = np.argmax(test_pred, axis=1)
            test_y_idx = np.asarray(test_y)
            res = evaluator.evaluate(test_pred_idx, test_y_idx)
            print(res)
    elif model_type == "dense":
        sess = tf.Session()
        x_ph, y_ph, is_train_ph, train_op, out = tf_dense()
        sess.run(tf.global_variables_initializer())

        iterator = BatchIterator(batch_size=args.batch_size)
        evaluator = AccEvaluator()
        for epoch in range(args.num_ep):
            t_start = time.time()
            for batch in iterator(train_x, train_y):
                sess.run(train_op, feed_dict={
                    x_ph: batch.inputs, y_ph: batch.targets, is_train_ph: True})
            print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
            # evaluate
            test_pred = sess.run(out, feed_dict={x_ph: test_x, is_train_ph: False})
            test_pred_idx = np.argmax(test_pred, axis=1)
            test_y_idx = np.asarray(test_y)
            res = evaluator.evaluate(test_pred_idx, test_y_idx)
            print(res)
    else:
        raise ValueError("Invalid model_type!")


def tf_cnn():
    with tf.variable_scope("cnn"):
        x = tf.placeholder(dtype=tf.float64, shape=[None, 28, 28, 1])
        y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
        is_train_ph = tf.placeholder(dtype=bool)
        conv1 = tf.layers.conv2d(x, 4, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 8, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        out = tf.layers.dense(fc1, 10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=out, labels=y))
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return x, y, is_train_ph, train_op, out


def tf_dense():
    with tf.variable_scope("dense"):
        x = tf.placeholder(dtype=tf.float64, shape=[None, 784])
        y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
        is_train_ph = tf.placeholder(dtype=bool)
        fc1 = tf.layers.dense(x, 200, activation=tf.nn.relu)
        fc1 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)
        fc1 = tf.layers.dense(fc1, 70, activation=tf.nn.relu)
        fc1 = tf.layers.dense(fc1, 30, activation=tf.nn.relu)
        out = tf.layers.dense(fc1, 10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=out, labels=y))
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return x, y, is_train_ph, train_op, out


def main(args):
    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    if args.model_type == "cnn":
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

    dataset = train_x, train_y, test_x, test_y
    run_tinynn(args.model_type, dataset)
    if args.tf:
        run_tf(args.model_type, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="cnn", type=str, help="cnn or dense")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--tf", default=False, type=bool)
    args = parser.parse_args()
    main(args)
