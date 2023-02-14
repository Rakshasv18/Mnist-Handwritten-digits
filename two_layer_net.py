#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:54:18 2022

@author: raksha
"""

import numpy as np
import gradients
import activation
import losses
from tqdm import tqdm
import pickle


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std\
                            *np.random.randn(input_size, hidden_size)
                            
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['w2'] = weight_init_std\
                            *np.random.randn(hidden_size, output_size)
                            
        self.params['b2'] = np.zeros(output_size)
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        
        
    def predict(self, x):
         w1, w2 = self.params['w1'], self.params['w2']
         b1, b2 = self.params['b1'], self.params['b2']
         
         a1 = np.dot(x, w1) + b1
         
         z1 = activation.sigmoid(a1)
         a2 = np.dot(z1, w2) + b2
         y = activation.softmax(a2)
         return y
     
    def loss(self, x, t):
        y = self.predict(x)
        return losses.cross_entropy_mb(y, t)
    
    
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {} 
        grads['w1'] = gradients.numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = gradients.numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = gradients.numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = gradients.numerical_gradient(loss_w, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads= {}
        
        batch_num = x.shape[0]
        
        #forward
        a1 = np.dot(x, w1) + b1
        z1 = activation.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = activation.softmax(a2)
        
        #backward
        dy = (y - t)/batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, w2.T)
        da1 = activation.sigmoid_grad(a1) * dz1
        grads['w1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        
        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis =1 )
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    
    def fit(self, interations, x_train, t_train, x_test, t_test, batch_size, \
            learning_rate=0.1, backprop=True):
        
        
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size/batch_size, 1)
    
        print("start training......")
        
        for i in tqdm(range(interations)):
            #get mini batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
        
            if backprop:
                grad = self.gradient(x_batch, t_batch)
            else:
                grad = self.numerical_gradient(x_batch, t_batch)
          
            
            for key in ('w1', 'b1', 'w2', 'b2'):
                self.params[key] -= learning_rate*grad[key]
                
            loss = self.loss(x_batch, t_batch)
            self.train_losses.append(loss)
            
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                self.train_accs.append(train_acc)
                self.test_accs.append(test_acc)
                
        print('Done')
                
            
            
    def save_model(self, model_filename):
        print("saving_model....")
        with open(model_filename, 'wb') as f:
            pickle.dump(self.params, f, -1)
            
        print("Done saving the model")
        
    def load_model(self, model_filename):
        print("Loading_model....")
        with open(model_filename, 'rb') as f:
            self.params = pickle.load(f)
        print("done")
        
        
            
            
                
    



        

    