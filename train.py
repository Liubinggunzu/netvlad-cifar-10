# -*- coding: utf-8 -*-

import tensorflow as tf

import netvlad
import train_utils
import os
import math



tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'directory to save trained models')
tf.app.flags.DEFINE_string('pretrained_model', 'vgg16.npy', 'pretrained netvlad model path')
tf.app.flags.DEFINE_string('train_record_path', 'records/train', 'path of training tfrecords file')
tf.app.flags.DEFINE_string('val_record_path', 'records/val', 'path of validation tfrecords file')
tf.app.flags.DEFINE_string('test_record_path', 'records/test', 'path of testing tfrecords file')

tf.app.flags.DEFINE_integer('batch_size', 120, 'num of triplets in a batch')
tf.app.flags.DEFINE_integer('numEpoch', 30, 'num of epochs to train')
tf.app.flags.DEFINE_integer('lr', 0.0001, 'initial learning rate')
tf.app.flags.DEFINE_integer('print_every', 5, 'print every ... batch')
tf.app.flags.DEFINE_integer('save_every', 5, 'save model every ... epochs')

FLAGS = tf.app.flags.FLAGS


def main(_):
    print('checkpoint 11111111')
    filenameQueue = train_utils.generate_filenamequeue(FLAGS.train_record_path, FLAGS.numEpoch)
    print('checkpoint 22222222')
    [imgs, labels] = train_utils.next_batch(filenameQueue, FLAGS.batch_size)
    print('checkpoint 33333333')
    numTrainImg = train_utils.num_Train_Img('trainImgList.txt')
    print("number of imgs in train dataset is %s\n" % numTrainImg)

    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.device('/gpu:1'):
        with tf.Session(config = config) as sess:
            train_mode = tf.placeholder(tf.bool, name = 'train_mode')

            model = netvlad.Netvlad(FLAGS.pretrained_model)
            model.build(imgs, train_mode)
            print("number of total parameters in the model is %d\n" % model.get_var_count())

            # loss = -tf.reduce_sum(tf.one_hot(labels, depth = 429) * tf.log(model.prob))
            loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, depth = 429), model.fc8)
            train = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(loss)

            sess.run(init)
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
  
            train_loss = 0
            count = 0.0

            print("training begins!\n")

            try:
                while not coord.should_stop():
                    count += 1
                    _, train_loss = sess.run([train, loss], feed_dict = {train_mode: True})
                    if count % FLAGS.print_every == 0:
                        print("Epoch: %.4f  training_loss = %.6f\n" % (count * FLAGS.batch_size / numTrainImg, train_loss))
                    if (count * FLAGS.batch_size) % (FLAGS.save_every * numTrainImg) == 0:
                        model.save_npy(sess, "%s/netvlad_epoch_%d_loss_%.6f" % (FLAGS.checkpoint_dir, (count * FLAGS.batch_size) / numTrainImg, train_loss))
            except tf.errors.OutOfRangeError:
                print("training finished")
            finally:
                coord.request_stop()
        
            coord.join(threads)
                
if __name__ == '__main__':
    tf.app.run()