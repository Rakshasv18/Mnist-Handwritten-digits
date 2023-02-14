#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:55:49 2022

@author: raksha
"""

import numpy as np
from two_layer_net import TwoLayerNet
from mnist import Mnist
from tqdm import tqdm
import matplotlib.pyplot as plt


trained_model_filename = 'varahamurthy_mnist_nn_model.pkl'

mnist = Mnist()
(x_train, t_train), (x_test, t_test) = mnist.load_data(normalize=True, one_hot_label=True)

plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.show()
print(t_train[0])


iterations = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
hidden_size = 50

net = TwoLayerNet(input_size= mnist.image_size, hidden_size=hidden_size, output_size=10)

net.fit(iterations, x_train, t_train, x_test, t_test, batch_size, learning_rate= learning_rate, backprop=True)


net.save_model(trained_model_filename)


#visualise loss

plt.figure()
x = np.arange(len(net.train_losses))
plt.plot(x, net.train_losses, label='train loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#visualize acc
plt.figure()
markers = {'train': '0', 'test': 's'}
x = np.arange(len(net.train_accs))
plt.plot(x, net.train_accs, label='train acc')

plt.plot(x, net.test_accs, label='test acc', linestyle='--')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()

