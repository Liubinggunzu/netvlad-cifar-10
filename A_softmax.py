import tensorflow as tf
import math
import numpy as np

# input x is B x W
# input y is B x H one-hot tensor
# input W_norm is W x H
# input fc is B x H
# fc = tf.matmul(x, W_norm)
def A_softmax(x, y, W_norm, fc, m, batch_size, numClass):
    print(fc)
    w_yi = tf.matmul(y, W_norm, transpose_b = True)     # w_yi is B x W
    f_yi = tf.reduce_sum(tf.multiply(fc, y), axis = -1)     # f_yi is B 

    w_norm = tf.norm(w_yi, axis = -1)   # w_norm is B 
    x_norm = tf.norm(x, axis = -1)      # x_norm is B

    cos_thelta = tf.divide(f_yi, w_norm * x_norm)   # cos_thelta is B
    phi_thelta, K = func_thelta(cos_thelta, m, batch_size)
    A = w_norm * x_norm * phi_thelta
    C = []
    for i in range(batch_size):
        D = []
        for j in range(numClass):
            D.append(tf.cond(tf.equal(1.0, y[i, j]), lambda: A[i], lambda: fc[i, j]))
        E = tf.stack(D)
        C.append(E)
            
    F = tf.stack(C)
    # print(D.get_shape())
    fc_softmax = tf.nn.softmax(F)
    loss = tf.reduce_sum(-tf.log(fc_softmax) * y)
    return loss, K

def func_thelta(cos_thelta, m, batch_size):
    if m == 2:
        cos_m_thelta = 2 * cos_thelta ** 2 - 1
    elif m == 3:
        cos_m_thelta = 4 * cos_thelta ** 3 - 3 * cos_thelta
    elif m == 4:
        cos_m_thelta = 8 * cos_thelta ** 4 - 8 * cos_thelta ** 2 + 1

    L = [math.cos(float(i + 1) / m * math.pi) for i in range(m)]
    L_constant = tf.constant(value = L)
    # K = tf.Variable(tf.zeros([batch_size]))
    k = [0.0] * batch_size
    # K = tf.Variable(initial_value = np.zeros((batch_size, )), trainable = False)
    for i in range(batch_size):
        for j in range(m):
            k[i] += tf.cast(tf.less_equal(cos_thelta[i], L_constant[j]), tf.float32)
    K = tf.stack(k)
    func_thelta = ((-1) ** K) * cos_m_thelta - 2 * K

    return func_thelta, K