import tensorflow as tf
import numpy as np

import pickle
import os

import os
import pickle
import numpy as np
import tensorflow as tf

def load_local_cifar10(dataset_dir):
    x_train, y_train = [], []
    
    for i in range(1, 6):
        batch_file = os.path.join(dataset_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            x_train.append(batch['data'])
            y_train.extend(batch['labels'])
            
    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    
    with open(os.path.join(dataset_dir, 'test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        x_test = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(batch['labels'])
        
    return (x_train, y_train), (x_test, y_test)

def preprocess(image, label):
    image = tf.image.resize(image, (227, 227))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label