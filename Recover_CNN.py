import tensorflow as tf
import numpy as np
import pandas as pd
from Import_Images import Import_Data
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# path = './Training_Data/'
path = './color1/'
# img_train, img_test, label_train, label_test = IMG(path)
# print(label_test)
data = Import_Data(path)

n_class = 4
batch_size = 8
# pixel = 784

# x = tf.placeholder('float', [None, pixel])
x = tf.placeholder('float')
y = tf.placeholder('int32')

def conv_2d(x, W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = "SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")

def model(x):
    
    weights = {"W_conv1": tf.Variable(tf.random_normal([5,5,3,32])), #CHANGES HERE
                "W_conv2": tf.Variable(tf.random_normal([5,5,32,64])),
                "W_full": tf.Variable(tf.random_normal([14*14*64,1000])),
                "W_out": tf.Variable(tf.random_normal([1000, n_class]))
    }

    bias = {"bias_conv1": tf.Variable(tf.random_normal([32])),
            "bias_conv2": tf.Variable(tf.random_normal([64])),
            "bias_full": tf.Variable(tf.random_normal([1000])),
            "bias_out": tf.Variable(tf.random_normal([n_class])),
    }

    x = tf.reshape(x, shape = [-1, 224, 224, 3])

    conv1 = tf.nn.relu(conv_2d(x, weights['W_conv1'])+bias['bias_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv_2d(conv1, weights['W_conv2'])+bias['bias_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, shape = [-1,14*14*64])
    full = tf.matmul(fc, weights['W_full']) + bias['bias_full']
    full = tf.nn.relu(full)

    output = tf.add(tf.matmul(full, weights['W_out']),bias['bias_out'])

    return output

def train_CNN(x):

    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    number_of_epoch = 10
    saver = tf.train.Saver()

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./Checkpoints/model.ckpt")

        for epoch in range(number_of_epoch):
            loss = 0
            for _ in range(int(data.num_examples/batch_size)):
                epoch_x,epoch_y = data.next_train_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                loss += c
            print(loss)

        # correct = tf.equal(tf.arg_max(prediction,1), tf.arg_max(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print("Training accuracy is:  ", accuracy.eval({x: data.img_test, y:data.label_test}))

train_CNN(x)