#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:53:09 2022

@author: raksha
"""

import numpy as np
import gzip
import urllib.request
import os
import pickle


class Mnist():
    image_size = 784
    image_dim = (1, 28, 28)
    train_num = 60000
    test_num = 10000
    
    mnist_pkl_filename = 'mnist_datasets.pkl'
    key_file = {
    'train_images' : 'train-images-idx3-ubyte.gz',
    'train_labels' : 'train-labels-idx1-ubyte.gz',
    'test_images' : 't10k-images-idx3-ubyte.gz',
    'test_labels' : 't10k-labels-idx1-ubyte.gz'
    }
    
    
    
    def __init__(self):
        self.network = None
        
    def _download_datasets(self):
        url_base = 'http://yann.lecun.com/exdb/mnist/'
        
        #download all mnist datasets
        
        for value in self.key_file.values():
            if os.path.exists(value):
                print('File exists.')
                
            else:
                print('Downloading {}.....'.format(value))
                urllib.request.urlretrieve(url_base + value, value)
                print('Done')
                
    def _load_images(self,file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.image_size)
        
        
        print('Done with loading images:', file_name)
        return images

    def _load_labels(self,file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print('Done with labelling the images:', file_name)
        return labels
    
    
    
    def _change_one_hot_label(self, x):
        t = np.zeros((x.size, 10))
        for idx, row in enumerate(t):
            row[x[idx]] = 1
            
        return t
    
    def init_mnist(self):
        self._download_datasets()
        dataset = {}
        dataset['train_images'] = self._load_images(self.key_file['train_images'])
        dataset['train_labels'] = self._load_labels(self.key_file['train_labels'])
        dataset['test_images'] = self._load_images(self.key_file['test_images'])
        dataset['test_labels'] = self._load_labels(self.key_file['test_labels'])
        
        print('creating a pickle for the data....')
        
        with open(self.mnist_pkl_filename, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done")
        
    def load_data(self, normalize=False, flatten=True, one_hot_label=False):
        if not os.path.exists(self.mnist_pkl_filename):
            self.init_mnist()
            
        with open(self.mnist_pkl_filename, 'rb') as f:
            dataset = pickle.load(f)
            
        if normalize:
            for key in ('train_images', 'test_images'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0
                
        if one_hot_label:
            dataset['train_labels'] = self._change_one_hot_label(dataset['train_labels'])
            dataset['test_labels'] = self._change_one_hot_label(dataset['test_labels'])

            
        if not flatten:
            for key in ('train_images', 'test_images'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
                
        return (dataset['train_images'], dataset['train_labels']), \
                (dataset['test_images'], dataset['test_labels'])
                
                
    
if __name__ == '__main__':
    mnist = Mnist()
    mnist.init_mnist()
     
            
            

    





































