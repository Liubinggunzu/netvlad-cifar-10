# -*- coding: utf-8 -*-

import tensorflow as tf

import netvlad
import netvlad_1
import eva_utils
import os
import math

tf.app.flags.DEFINE_string('modelPath', 'checkpoint/', 'path of trained model')
tf.app.flags.DEFINE_integer('batch_size', 100, 'num of triplets in a batch')
tf.app.flags.DEFINE_integer('print_every', 5, 'print every ... batch')

tf.app.flags.DEFINE_boolean('use_vlad', True, 'use vlad pooling or not')

FLAGS = tf.app.flags.FLAGS


def main(_):
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.device('/gpu:0'):
        with tf.Session(config = config) as sess:
            X = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'X')
            Y = tf.placeholder(tf.int64, [None], name = 'Y')

            if FLAGS.use_vlad:
                model = netvlad_1.Netvlad(FLAGS.modelPath)
            else:
                model = netvlad.Netvlad(FLAGS.modelPath)
            model.build(X)
            print("number of total parameters in the model is %d\n" % model.get_var_count())

            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(model.fc3), axis = 1), Y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            sess.run(tf.global_variables_initializer())

            Acc = 0.0
            acc = 0.0
            numBatch = 10000 / FLAGS.batch_size

            print("evaluation begins!\n")
            count = 0.0
            for x, y in eva_utils.next_batch(FLAGS.batch_size, 'cifar-10-batches-py'):
                count += 1
                acc = sess.run(accuracy, feed_dict = {X: x, Y: y})
                Acc += acc
                if count % FLAGS.print_every == 0:
                    print("progress: %.4f      current accuracy = %.6f      total accuracy = %.6f\n" % (count / numBatch, acc, Acc / count))

if __name__ == '__main__':
    tf.app.run()