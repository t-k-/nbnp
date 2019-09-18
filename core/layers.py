"""Network layers and activation layers."""

import numpy as np

from core.initializer import UniformInit
from core.initializer import XavierUniformInit
from core.initializer import ZerosInit


class Layer(object):

    def __init__(self, name):
        self.name = name

        self.params, self.grads = {}, {}
        self.is_training = True

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == "TRAIN" else False


class Dense(Layer):

    def __init__(self,
                 num_out,
                 num_in=None,
                 w_init=XavierUniformInit(),
                 b_init=ZerosInit()):
        super().__init__("Linear")
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [num_in, num_out], "b": [1, num_out]}
        self.params = {"w": None, "b": None}

        self.is_init = False
        if num_in is not None:
            self._init_parameters(num_in)

        self.inputs = None

    def forward(self, inputs):
        # lazy initialize
        if not self.is_init:
            self._init_parameters(inputs.shape[1])

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T

    def _init_parameters(self, input_size):
        self.shapes["w"][0] = input_size
        self.params["w"] = self.initializers["w"](shape=self.shapes["w"])
        self.params["b"] = self.initializers["b"](shape=self.shapes["b"])
        self.is_init = True


class Conv2D(Layer):

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniformInit(),
                 b_init=ZerosInit()):
        """
        Implement 2D convolution layer
        :param kernel: A list/tuple of int that has length 4 (height, width, in_channels, out_channels)
        :param stride: A list/tuple of int that has length 2 (height, width)
        :param padding: String ["SAME", "VALID"]
        :param w_init: weight initializer
        :param b_init: bias initializer
        """
        super().__init__("Conv2D")

        # verify arguments
        assert len(kernel) == 4
        assert len(stride) == 2
        assert padding in ("SAME", "VALID")

        self.padding_mode = padding
        self.kernel = kernel
        self.stride = stride
        self.initializers = {"w": w_init, "b": b_init}

        self.is_init = False

        self.inputs = None
        self.cache = None  # cache

    def forward(self, inputs):
        if not self.is_init:
            self._init_parameters()

        k_h, k_w = self.kernel[:2]  # kernel size
        s_h, s_w = self.stride

        pad = self._get_padding([k_h, k_w], self.padding_mode)

        pad_width = ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0))
        padded = np.pad(inputs, pad_width=pad_width, mode="constant")
        pad_h, pad_w = padded.shape[1:3]

        in_n, in_h, in_w, in_c = inputs.shape
        out_h = int((in_h + pad[0] + pad[1] - k_h) / s_h + 1)
        out_w = int((in_w + pad[2] + pad[3] - k_w) / s_w + 1)
        out_c = self.kernel[-1]

        kernel = self.params["w"]
        col_len = np.prod(kernel.shape[:3])
        col_patches = list()
        for i, col in enumerate(range(0, pad_h - k_h + 1, s_h)):
            row_patches = list()
            for j, row in enumerate(range(0, pad_w - k_w + 1, s_w)):
                patch = padded[:, col:col+k_h, row:row+k_w, :]
                row_patches.append(patch)
            col_patches.append(row_patches)
        # shape of X_matrix [in_n * out_h * out_w, in_h * in_w * in_c]
        X_matrix = np.asarray(col_patches).reshape(
            (out_h, out_w, in_n, col_len)).transpose(
            [2, 0, 1, 3]).reshape((-1, col_len))

        # shape of W_matrix [in_h * in_w * in_c, out_c]
        W_matrix = kernel.reshape((col_len, -1))
        # outputs = X_matrix @ W_matrix
        outputs = (X_matrix @ W_matrix).reshape((in_n, out_h, out_w, out_c))
        self.cache = {"in_n": in_n, "in_img_size": (in_h, in_w, in_c),
                      "kernel_size": (k_h, k_w, in_c), "stride": (s_h, s_w),
                      "pad": pad, "pad_img_size": (pad_h, pad_w, in_c),
                      "out_img_size": (out_h, out_w, out_c),
                      "X_matrix": X_matrix, "W_matrix": W_matrix}
        # add bias
        outputs += self.params["b"]
        return outputs

    def backward(self, grad):
        in_n = self.cache["in_n"]
        in_h, in_w, in_c = self.cache["in_img_size"]
        k_h, k_w, _ = self.cache["kernel_size"]
        s_h, s_w = self.cache["stride"]
        out_h, out_w, out_c = self.cache["out_img_size"]
        pad_h, pad_w, _ = self.cache["pad_img_size"]
        pad = self.cache["pad"]

        d_w = self.cache["X_matrix"].T @ grad.reshape((-1, out_c))
        self.grads["w"] = d_w.reshape(self.params["w"].shape)
        self.grads["b"] = np.sum(grad, axis=(0, 1, 2))

        d_X_matrix = grad @ self.cache["W_matrix"].T
        d_in = np.zeros(shape=(in_n, pad_h, pad_w, in_c))
        for i, col in enumerate(range(0, pad_h - k_h + 1, s_h)):
            for j, row in enumerate(range(0, pad_w - k_w + 1, s_w)):
                patch = d_X_matrix[:, i, j, :].reshape(
                    (in_n, k_h, k_w, in_c))
                d_in[:, col:col+k_h, row:row+k_w, :] += patch
        # cut off padding
        d_in = d_in[:, pad[0]:pad_h-pad[1], pad[2]:pad_w-pad[3], :]
        return d_in

    @staticmethod
    def _get_padding(ks, mode):
        """
        params: ks (kernel size) [p, q]
        return: list of padding (top, bottom, left, right) in different modes
        """
        pad = None
        if mode == "FULL":
            pad = [ks[0] - 1, ks[1] - 1, ks[0] - 1, ks[1] - 1]
        elif mode == "VALID":
            pad = [0, 0, 0, 0]
        elif mode == "SAME":
            pad = [(ks[0] - 1) // 2, (ks[0] - 1) // 2,
                   (ks[1] - 1) // 2, (ks[1] - 1) // 2]
            if ks[0] % 2 == 0:
                pad[1] += 1
            if ks[1] % 2 == 0:
                pad[3] += 1
        else:
            print("Invalid mode")
        return pad

    def _init_parameters(self):
        self.params["w"] = self.initializers["w"](self.kernel)
        self.params["b"] = self.initializers["b"](self.kernel[-1])
        self.is_init = True


class MaxPool2D(Layer):

    def __init__(self,
                 pool_size,
                 stride,
                 padding="VALID"):
        """
        Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__("MaxPooling2D")
        # validate arguments
        assert len(pool_size) == 2
        assert len(stride) == 2
        assert padding in ("VALID", "SAME")

        self.pool_size = pool_size
        self.stride = stride
        self.padding_mode = padding

        self.cache = None

    def forward(self, inputs):
        in_n, in_h, in_w, in_c = inputs.shape
        pad = self._get_padding([in_h, in_w], self.pool_size, self.stride,
                                self.padding_mode)
        pad_width = ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0))
        padded = np.pad(inputs, pad_width=pad_width, mode="constant")
        pad_h, pad_w = padded.shape[1:3]
        (s_h, s_w), (pool_h, pool_w) = self.stride, self.pool_size
        out_h, out_w = pad_h // s_h, pad_w // s_w
        patches_list, max_pos_list = list(), list()
        for col in range(0, pad_h, s_h):
            for row in range(0, pad_w, s_w):
                pool = padded[:, col:col + pool_h, row:row + pool_w, :]
                pool = pool.reshape((in_n, -1, in_c))
                max_pos_list.append(np.argmax(pool, axis=1))
                patches_list.append(np.max(pool, axis=1))
        outputs = np.array(patches_list).transpose((1, 0, 2)).reshape(
            (in_n, out_h, out_w, in_c))
        max_pos = np.array(max_pos_list).transpose((1, 0, 2)).reshape(
            (in_n, out_h, out_w, in_c))

        self.cache = {"in_n": in_n, "in_img_size": (in_h, in_w, in_c),
                      "stride": (s_h, s_w), "pad": (pad_h, pad_w),
                      "pool": (pool_h, pool_w), "max_pos": max_pos,
                      "out_img_size": (out_h, out_w, in_c)}
        return outputs

    def backward(self, grad):
        in_n, (in_h, in_w, in_c) = self.cache["in_n"], self.cache["in_img_size"]
        s_h, s_w = self.cache["stride"]
        pad_h, pad_w = self.cache["pad"]
        pool_h, pool_w = self.cache["pool"]

        d_in = np.zeros(shape=(in_n, pad_h, pad_w, in_c))
        for i, col in enumerate(range(0, pad_h, s_h)):
            for j, row in enumerate(range(0, pad_w, s_w)):
                _max_pos = self.cache["max_pos"][:, i, j, :]
                _grad = grad[:, i, j, :]
                mask = np.eye(pool_h * pool_w)[_max_pos].transpose((0, 2, 1))
                region = np.repeat(_grad[:, np.newaxis, :],
                                   pool_h * pool_w, axis=1) * mask
                region = region.reshape((in_n, pool_h, pool_w, in_c))
                d_in[:, col:col + pool_h, row:row + pool_w, :] = region
        return d_in

    def _get_padding(self, input_size, pool_size, stride, mode):
        h_pad = self._get_padding_1d(
            input_size[0], pool_size[0], stride[0], mode)
        w_pad = self._get_padding_1d(
            input_size[1], pool_size[1], stride[1], mode)
        return h_pad + w_pad

    @staticmethod
    def _get_padding_1d(input_size, pool_size, stride, mode):
        if mode == "SAME":
            r = input_size % stride
            if r == 0:
                n_pad = max(pool_size - stride, 0)
            else:
                n_pad = max(pool_size, stride) - r
        else:
            n_pad = 0
        half = n_pad // 2
        pad = [half, half] if n_pad % 2 == 0 else [half, half + 1]
        return pad


class Flatten(Layer):

    def __init__(self):
        super().__init__("Flatten")
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.ravel().reshape(inputs.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__("Dropout")
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        if self.is_training:
            multiplier = np.random.binomial(
                1, self._keep_prob, size=inputs.shape)
            self._multiplier = multiplier / self._keep_prob
            outputs = inputs * self._multiplier
        else:
            outputs = inputs
        return outputs

    def backward(self, grad):
        assert self.is_training is True
        return grad * self._multiplier


class Activation(Layer):

    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("Sigmoid")

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1.0 - self.func(x))


class Softplus(Activation):

    def __init__(self):
        super().__init__("Softplus")

    def func(self, x):
        return np.log(1.0 + np.exp(x))

    def derivative_func(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class Tanh(Activation):

    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return np.tanh(x)

    def derivative_func(self, x):
        return 1.0 - self.func(x) ** 2


class ReLU(Activation):

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        return x > 0.0


class LeakyReLU(Activation):

    def __init__(self, slope=0.01):
        super().__init__("LeakyReLU")
        self._slope = slope

    def func(self, x):
        # TODO: maybe a litter bit slow due to the copy
        x = x.copy()
        x[x < 0.0] *= self._slope
        return x

    def derivative_func(self, x):
        x[x < 0.0] = self._slope
        return x


class RBM(Layer):

    def _init_parameters(self, n_visible):
        self.shape[0] = n_visible
        self.params["w"] = self.initializers["w"](shape=self.shape)
        self.params["v_b"] = self.initializers["v_b"](shape=n_visible)
        self.params["h_b"] = self.initializers["h_b"](shape=self.shape[1])

    def __init__(self,
                 n_hidden,
                 k=1, # sample times
                 n_visible=None,
                 sigmoid=Sigmoid().func,
                 softplus=Softplus().func,
                 w_init=UniformInit(),
                 v_b_init=UniformInit(),
                 h_b_init=UniformInit()):
        super().__init__("RBM")
        self.initializers = {"w": w_init, "v_b": v_b_init, "h_b": h_b_init}
        self.shape = [n_visible, n_hidden]
        self.params = {"w": None, "v_b": None, "h_b": None}
        self.k = k
        self.sigmoid = sigmoid;
        self.softplus = softplus;
        self.relu = ReLU().func

    def sample_by_prob(self, prob):
        rand = np.random.uniform(0.0, 1.0, prob.shape)
        sign = np.sign(prob - rand) # 1 or -1
        return self.relu(sign) # 1 or 0

    def v_to_h(self, v):
        p_h = self.sigmoid(v @ self.params['w'] + self.params['h_b'])
        sample_h = self.sample_by_prob(p_h)
        return sample_h

    def h_to_v(self, h):
        p_v = self.sigmoid(h @ self.params['w'].T + self.params['v_b'])
        sample_v = self.sample_by_prob(p_v)
        return sample_v

    def gibs_sampling(self, v):
        # lazy initialize
        if self.shape[0] is None:
            self._init_parameters(v.shape[1])

        # gibs sampling (k times)
        self.v_0 = v[0,:]
        for _ in range(self.k):
            h = self.v_to_h(v)
            v = self.h_to_v(h)
        self.v_k = v[-1,:]
        return v

    def step(self, lr):
        p_h_0 = self.sigmoid(self.v_0 @ self.params['w'] + self.params['h_b'])
        p_h_k = self.sigmoid(self.v_k @ self.params['w'] + self.params['h_b'])
        p_h_0_ = np.asarray([p_h_0])
        p_h_k_ = np.asarray([p_h_k])
        v_0 = np.asarray([self.v_0]).T
        v_k = np.asarray([self.v_k]).T
        # update parameters
        self.params['w'] += lr * (v_0 @ p_h_0_ - v_k @ p_h_k_)
        self.params['v_b'] += lr * (self.v_0 - self.v_k)
        self.params['h_b'] += lr * (p_h_0 - p_h_k)
        return 0
