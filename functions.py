import numpy as np
import os.path
import gzip


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def relu(x):
    return np.maximum(x, 0)


# def softmax(x):
#     c = np.max(x)
#     exp_x = np.exp(x - c)
#     sum_exp_x = np.sum(exp_x)
#     return exp_x / sum_exp_x
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # ラベル表現
    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + delta)) / batch_size
    # one-hot表現
    # return -np.sum(np.log(y[np.arange(batch_size), t]+delta))/batch_size

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
#     if t.size == y.size:
#         t = t.argmax(axis=1)

#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        original_x_i = x[i]

        # f(x+h)
        x[i] = original_x_i + h
        f_x_plus_h = f(x)

        # f(x-h)
        x[i] = original_x_i - h
        f_x_minus_h = f(x)

        grad[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)
        x[i] = original_x_i  # x[i] をもとに戻す

    return grad


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# print(numerical_gradient(function_2, np.array([3.0, 4.0]))
# print('hello')
