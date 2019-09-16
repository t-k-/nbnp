"""Example code for MNIST. A fully-connected network and a convolutional neural network were implemented."""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.evaluator import AccEvaluator
from core.layers import Conv2D
from core.layers import MaxPool2D
from core.layers import Dense
from core.layers import Flatten
from core.layers import ReLU
from core.layers import Tanh
from core.losses import MSELoss, MAELoss
from core.losses import SoftmaxCrossEntropyLoss
from core.model import AutoEncoder
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    # load the dataset
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

from matplotlib import pyplot as plt
from matplotlib import cm as cm
def disp(batch, fig=None):
    batch = batch[:]
    batch.resize(28, 28)
    if fig is None:
        fig = plt.figure()
        fig.show()
    ax = fig.gca()
    ax.imshow(batch, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    fig.canvas.draw()
    return fig

import scipy.ndimage as nd
def gaussian_blur(img, sigma):
    img = nd.filters.gaussian_filter(img, sigma, order=0)
    return img

def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    if args.model_type == "cnn":
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

    if args.model_type == "cnn":
        net = Net([
            Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1], padding="SAME"), # (1, 28, 28, 6)
            ReLU(), # (1, 28, 28, 6)
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]), # (1, 14, 14, 6)
            Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1], padding="SAME"), # (1, 14, 14, 16)
            ReLU(), #(1, 14, 14, 16)
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]), # (1, 7, 7, 16)
            Flatten(), # (1, 784)
            Dense(120), # (1, 120)
            ReLU(), # (1, 120)
            Dense(84), # (1, 84)
            ReLU(), # (1, 84)
            Dense(10) # (1, 10)
        ])
    elif args.model_type == "dense":
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

    # specify the encoder and decoder net structure
    encoder = Net([
        Dense(256),
        ReLU(),
        Dense(64)
    ])

    decoder = Net([
        ReLU(),
        Dense(256),
        Tanh(),
        Dense(784),
        Tanh()
    ])

    dae = AutoEncoder(encoder=encoder, decoder=decoder, loss=MSELoss(), optimizer=Adam(args.lr))
    dae.load('output/model.pkl')

    model = Model(net=net, loss=SoftmaxCrossEntropyLoss(), optimizer=Adam(lr=args.lr))
    model.load('lenet-relu.pkl')
    for i in range(16):
        print('prefer', i)
        prefer(args, model, i)

def prefer(args, model, choose):
    img_mean = 0.456
    img_std = 0.224
    iter_n = 200
    sigma_start, sigma_end = 0.35, 0.30

    opt = Adam(lr=args.lr)
    max_img = np.random.normal(img_mean, img_std, (1, 28, 28, 1))

    fig = disp(max_img)

    # grad_fix = np.zeros((1, 10))
    # grad_fix[0][choose] = - 1.0

    grad_fix = np.zeros((1, 14, 14, 16))
    grad_fix[:,:,:,choose] = -1

    for epoch in range(iter_n):
        # forward
        inputs = max_img
        for layer in model.net.layers[0:4]:
            inputs = layer.forward(inputs)
        # backward
        grad = grad_fix
        for layer in model.net.layers[0:4][::-1]:
            grad = layer.backward(grad)
        # update image
        flat_grad = np.ravel(grad)
        flat_step = opt._compute_step(flat_grad)
        step = flat_step.reshape(max_img.shape)
        max_img += step
        # blur image
        sigma = sigma_start + epoch * (sigma_end - sigma_start) / iter_n
        # max_img = gaussian_blur(max_img, sigma)
        # make image range valid
        cur_mean = max_img.mean()
        cur_std = max_img.std()
        cur_max = max_img.max()
        cur_min = max_img.min()
        max_img = (max_img - cur_min) / (cur_max - cur_min)

        cur_mean = max_img.mean()
        cur_std = max_img.std()
        cur_max = max_img.max()
        cur_min = max_img.min()

        if epoch % 100 == 0:
            loss = - inputs[:,:,:,choose].sum()
            disp(max_img, fig)
            print('#%d: Loss: %.5f' % (epoch, loss), end=" ")
            print('u=%.3f, std=%.3f, range=(%.3f, %.3f)' %
                (cur_mean, cur_std, cur_min, cur_max), end=" ")
            print('sigma: %.2f' % sigma)

    print('done')
    disp(max_img, fig)
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="cnn", type=str, help="cnn or dense")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
