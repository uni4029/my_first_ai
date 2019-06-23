import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
import pickle
import functions as f

use_batch = True


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return (x_train, t_train), (x_test, t_test)


def init_network():
    with open("weight.pkl", 'rb') as file:
        network = pickle.load(file)

    return network


def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    a1 = np.dot(x, W1) + b1
    z1 = f.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = f.softmax(a2)

    return y


(x_train, t_train), (x_test, t_test) = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
if use_batch:
    for i in range(0, len(x_test), batch_size):
        x_test_batch = x_test[i:i + batch_size]
        y_test_batch = predict(network, x_test_batch)
        p = np.argmax(y_test_batch, axis=1)
        accuracy_cnt += np.sum(p == t_test[i:i + batch_size])
else:
    for i in range(len(x_test)):
        y = predict(network, x_test[i])
        p = np.argmax(y)
        if p == t_test[i]:
            accuracy_cnt += 1


print('accuracy' + str(float(accuracy_cnt) / len(x_test)))
