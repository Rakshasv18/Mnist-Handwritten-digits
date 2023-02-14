#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:56:08 2022

@author: raksha
"""


import numpy as np
from two_layer_net import TwoLayerNet
from mnist import Mnist
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image



mnist = Mnist()
(x_train, t_train), (x_test, t_test) = mnist.load_data(normalize=True, one_hot_label=True)

plt.imshow(x_test[1].reshape(28, 28), cmap='gray')
plt.show()
print(t_test[1])


iterations = 1000
train_size = x_train.shape[1]
batch_size = 100
learning_rate = 0.1
hidden_size = 50

net = TwoLayerNet(input_size= mnist.image_size, hidden_size=hidden_size, output_size=10)

#load the model
loaded_model = net.load_model("varahamurthy_mnist_nn_model.pkl")

print("model is loaded.....")

#predict on test data

img_size = 784           #28x28
img_dim = (1, 28, 28)
def image_preprocess(img_path):
    img = Image.open(img_path)
    img = img.resize((img_dim[1], img_dim[2]))
    img = img.convert("L")
    img = np.array(img)
    img = img.flatten()
    img_pp = (img.astype(np.float32) / 127.0) - 1
    return img_pp



rows, col = 10,5
for i in range(10):
    file = '/Users/raksha/Downloads/Pattern_Recognition_NN/assignment_6_7/digits/'+str(i)+'/*.png'
    get_imgs = glob.glob(file)
    images = [cv2.imread(image) for image in glob.glob(file)]
    fig = plt.figure(figsize = (30,30))
    for j in range(5):
        ax = fig.add_subplot(1, 5, j+1)
        plt.imshow(images[j])
        im_path = get_imgs[j]
        img = image_preprocess(im_path)
        yhat = np.argmax(net.predict(img))
        confidence = np.max(net.predict(img))
        title = 'Prediction:{} & Confidence:{}'.format(yhat,confidence)
        ax.title.set_text(title)


