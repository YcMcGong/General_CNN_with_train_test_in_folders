"""
Copyright (c) 2017 Yicong Gong <gongyc2@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from Import_Images import Import_Data

"""
This is the main program to train the CNN

"""
#Define values
path = './Test_Data/'
network_path = './CP_BK/1000/Checkpoints/'
# network_path = './Checkpoints/'
n_class = 8
sample_size = 4
batch_size = 32
subject_list = ['Apple','Car','Cow','Cup','Dog','Horse','Pear','Tomato']
final_layer_size = 1000
#Import Data
data = Import_Data(path,sample_size,subject_list)

#Set up tensorflow placeholder variable
x = tf.placeholder('float')
y = tf.placeholder('int64')

#Define Convolution function
def conv_2d(x, W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = "SAME")

#Define Pooling function
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")

#Define Model
def model(x):
    
    weights = {"W_conv1": tf.Variable(tf.random_normal([5,5,3,32])), #CHANGES HERE
                "W_conv2": tf.Variable(tf.random_normal([5,5,32,64])),
                "W_full": tf.Variable(tf.random_normal([14*14*64,final_layer_size])),
                "W_out": tf.Variable(tf.random_normal([final_layer_size, n_class]))
    }

    bias = {"bias_conv1": tf.Variable(tf.random_normal([32])),
            "bias_conv2": tf.Variable(tf.random_normal([64])),
            "bias_full": tf.Variable(tf.random_normal([final_layer_size])),
            "bias_out": tf.Variable(tf.random_normal([n_class])),
    }

    x = tf.reshape(x, shape = [-1, 224, 224, 3])

    conv1 = tf.nn.relu(conv_2d(x, weights['W_conv1'])+bias['bias_conv1'])
    conv1_pool = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv_2d(conv1_pool, weights['W_conv2'])+bias['bias_conv2'])
    conv2_pool = maxpool2d(conv2)

    fc = tf.reshape(conv2_pool, shape = [-1,14*14*64])
    full = tf.matmul(fc, weights['W_full']) + bias['bias_full']
    full = tf.nn.relu(full)

    output = tf.add(tf.matmul(full, weights['W_out']),bias['bias_out'])

    return output

#Run CNN
def test_CNN(x):

    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Tensorflow saver object declared
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, network_path + "model.ckpt") # Recover the trained network
 
        # Test
        temporary_accuracy = 0
        batch_count = int(data.num_images/batch_size)

        # Run Batches
        for _ in range(batch_count):
            epoch_x,epoch_y = data.next_img_batch(batch_size)
            correct = tf.equal(tf.arg_max(prediction,1), y)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            temporary_accuracy = temporary_accuracy + accuracy.eval(feed_dict = {x: epoch_x, y:epoch_y})

        # Add up accuracy
        test_accuracy = (temporary_accuracy/batch_count)

        print("Test Accuracy is ", test_accuracy)

# call main
if __name__ == '__main__':
    test_CNN(x)
