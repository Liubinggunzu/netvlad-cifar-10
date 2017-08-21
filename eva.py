# -*- coding: utf-8 -*-

import tensorflow as tf

import netvlad
import eva_utils
import os
import math


tf.app.flags.DEFINE_integer('batch_size', 100, 'num of triplets in a batch')
tf.app.flags.DEFINE_integer('print_every', 5, 'print every ... batch')

FLAGS = tf.app.flags.FLAGS


def main(_):
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.device('/gpu:1'):
        with tf.Session(config = config) as sess:
            X = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'X')
            Y = tf.placeholder(tf.int32, [None], name = 'Y')
            train_mode = tf.placeholder(tf.bool, name = 'train_mode')

            model = netvlad.Netvlad('checkpoint/epoch_25_loss_0.338908.npy')
            model.build(X, train_mode)
            print("number of total parameters in the model is %d\n" % model.get_var_count())

            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(model.fc2), axis = 1), Y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            sess.run(tf.global_variables_initializer())

            Acc = 0.0
            numBatch = 10000 / FLAGS.batch_size

            print("evaluation begins!\n")
            count = 0.0
            for x, y in eva_utils.next_batch(FLAGS.batch_size, 'cifar-10-batches-py'):
                count += 1
                accuracy = sess.run([accuracy], feed_dict = {X: x, Y: y, train_mode: False})
                Acc += accuracy
                if count % FLAGS.print_every == 0:
                    print("progress: %.4f      current accuracy = %.6f      total accuracy = %.6f\n" % (count / numBatch, accuracy, Acc / count))

if __name__ == '__main__':
    tf.app.run()