import tensorflow as tf
import numpy as np
import pandas as pd
from Import_Images import Import_Data

"""
This is the main program to train the CNN

"""
#Define values
path = './Training_Data/'
n_class = 8
sample_size = 600
batch_size = 432#90*4
validate_batch_size = int(sample_size*0.2)
number_of_epoch = 10
subject_list = ['Apple','Car','Cow','Cup','Dog','Horse','Pear','Tomato']
final_layer_size = 1500

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
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv_2d(conv1, weights['W_conv2'])+bias['bias_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, shape = [-1,14*14*64])
    full = tf.matmul(fc, weights['W_full']) + bias['bias_full']
    full = tf.nn.relu(full)

    output = tf.add(tf.matmul(full, weights['W_out']),bias['bias_out'])

    return output

#Define a validation function
def validate(validate_batch_size, prediction):
    
    validate_accuracy = 0
    validate_batch_count = int(data.num_validate_examples/validate_batch_size)
    
    for _ in range(validate_batch_count):
        epoch_x,epoch_y = data.next_validate_batch(validate_batch_size)
        correct = tf.equal(tf.arg_max(prediction,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        validate_accuracy = validate_accuracy + accuracy.eval(feed_dict = {x: epoch_x, y:epoch_y})
    return (validate_accuracy/validate_batch_count)

#Run CNN
def train_CNN(x):

    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

    #Tensorflow saver object declared
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./Checkpoints/model.ckpt") # Recover the trained network
        for epoch in range(number_of_epoch):
            loss = 0
            for _ in range(int(data.num_examples/batch_size)):
                epoch_x,epoch_y = data.next_train_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                loss += c
            saver.save(sess, './Checkpoints/model.ckpt') #Save the current network in checkpoint
            #data.Shuffule()
            print("Loss for epoch ", epoch, " is: ",loss)

            #Print out the validation result every 5 epoch
            if (epoch%5 ==0): 
                print("Validate Accuracy is ", validate(validate_batch_size, prediction))

        # correct = tf.equal(tf.arg_max(prediction,1), tf.arg_max(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print("Training accuracy is:  ", accuracy.eval({x: data.img_test, y:data.label_test}))

# call main
if __name__ == '__main__':
    train_CNN(x)