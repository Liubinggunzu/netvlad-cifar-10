import os
import math
import cPickle
import numpy as np

def next_batch(batch_size, dataPath):
    x = np.zeros((batch_size, 32, 32, 3))
    y = np.zeros((batch_size, ))
    with open('%s/test_batch' % (dataPath), 'rb') as fo:
        dic = cPickle.load(fo)
    X = dic['data']
    X = X / 255.0
    Y = dic['labels']
    numBatches = 10000 / batch_size
    for j in range(numBatches):
        X_batch = X[j * batch_size : (j + 1) * batch_size, :]
        Y_batch = Y[j * batch_size : (j + 1) * batch_size]
        y = np.array(Y_batch)
        x[:, :, :, 0] = X_batch[:, 0:1024].reshape((batch_size, 32, 32))
        x[:, :, :, 1] = X_batch[:, 1024:2048].reshape((batch_size, 32, 32))
        x[:, :, :, 2] = X_batch[:, 2048:3072].reshape((batch_size, 32, 32))

        yield x, y
            
    return