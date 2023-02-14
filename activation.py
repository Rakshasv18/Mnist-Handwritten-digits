#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:53:38 2022

@author: raksha
"""

import numpy as np


def sigmoid(a):
    return 1/(1+ np.exp(-a))


def sigmoid_grad(x):
    x = x - np.max(x, axis = -1, keepdims=True)
    return (1.0 - sigmoid((x)) * sigmoid(x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)


def softmax_1d(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y
    
    






