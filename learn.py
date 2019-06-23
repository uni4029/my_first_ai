import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
import pickle
import functions as f
import twolayernet

# use_batch = True


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, flatten=True, one_hot_label=True)


# ハイパーパラメータ
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size/batch_size, 1)

network = twolayernet.TwoLayerNet()

for i in range(iter_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_train_batch, t_train_batch)
    grad = network.gradient(x_train_batch, t_train_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_train_batch, t_train_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc:' + str(train_acc)+'    ,test acc:' + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

with open('weight.pkl', 'w+b') as file:
    pickle.dump(network.params, file)
    file.close()
    print('重みとバイアスを保存しました!')
