import os

import numpy as np
import tensorflow as tf


class Netvlad:
    def __init__(self, npy_path = None, trainable = True):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding = 'latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb):
        """
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        self.conv1 = self.conv_layer(rgb, 3, 64, "conv1")
        # self.norm1 = tf.nn.lrn(self.conv1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        self.pool1 = self.max_pool(self.conv1, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 64, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 64, 64, "conv2_2")
        self.conv2_3 = self.conv_layer(self.conv2_2, 64, 64, "conv2_3")
        self.conv2_4 = self.conv_layer(self.conv2_3, 64, 64, "conv2_4")
        # self.norm2 = tf.nn.lrn(self.conv2_4, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        self.pool2 = self.max_pool(self.conv2_4, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 64, 128, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 128, 128, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 128, 128, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 128, 128, "conv3_4")
        # self.norm3 = tf.nn.lrn(self.conv3_4, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 128, 256, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 256, 256, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 256, 256, "conv4_3")
        self.conv4_4 = self.conv_layer_last(self.conv4_3, 256, 256, "conv4_4")

        self.vlad_output = self.vlad_pooling_layer(self.conv4_4, 16, 100, 'vlad_pooling')

        self.fc1 = self.fc_layer(self.vlad_output, 4096, 384, 'fc1')

        self.fc2 = self.fc_layer(self.fc1, 384, 192, 'fc2')

        self.fc3 = self.fc_layer(self.fc2, 192, 10, 'fc3')

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
    
    def conv_layer_last(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def vlad_pooling_layer(self, bottom, k_cluster, alpha, name):
        with tf.variable_scope(name):
            filt, conv_biases, centers = self.get_vald_pooling_var(k_cluster, alpha, name)

            conv_reshape = tf.reshape(bottom, shape = [-1, (bottom.get_shape().as_list()[1] * bottom.get_shape().as_list()[2]), 256], name = 'reshape')    # conv_reshape is B x N x D
            conv_norm = tf.nn.l2_normalize(conv_reshape, dim = 1)
            descriptor = tf.expand_dims(conv_norm, axis = -1, name = 'expanddim')  # descriptor is B x N x D x 1
            conv_vlad = tf.nn.convolution(descriptor, filt, padding = 'VALID')  # conv_vlad is B x N x 1 x K
            bias = tf.nn.bias_add(conv_vlad, conv_biases)
            a_k = tf.nn.softmax(tf.squeeze(bias, axis = 2), dim = -1, name = "vlad_softmax")     # a_k is B x N x K

            V1 = tf.matmul(conv_reshape, a_k, transpose_a = True)    # V_1 is B x D x K
            V2 = tf.multiply(tf.reduce_sum(a_k, axis = 1, keep_dims = True), centers)     # V_1 is B x D x K
            V = tf.subtract(V1, V2)

            norm = tf.nn.l2_normalize(tf.reshape(tf.nn.l2_normalize(V, dim = 1), shape = [-1, 4096]), dim = 1)     # norm is B x (D x K)

            return norm

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.05)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, .01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases


    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.05)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, .01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_vald_pooling_var(self, k_cluster, alpha, name):
        initial_value = tf.truncated_normal([1, 256, 1, k_cluster], 0.0, 0.05)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([256, k_cluster], 0.0, 0.05)
        centers = self.get_var(initial_value, name, 1, name + '_centers')

        initial_value = tf.truncated_normal([k_cluster], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 2, name + '_biases')

        return filters, biases, centers

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name = var_name)
        else:
            var = tf.constant(value, dtype = tf.float32, name = var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path = "./netvlad-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
