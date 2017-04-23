import tensorflow as tf
import numpy as np
import pandas as pd
from Import_Images import Import_Data
from sklearn.decomposition import PCA
from scipy.stats import variation as sv
from matplotlib import pyplot as plt

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

    # return output
    return output, conv1, conv1_pool, conv2, conv2_pool, full

def plot_hist_of_layer(layer):
        layer_being_process = layer
        layer_being_process = np.reshape(layer_being_process, (len(layer_being_process),-1)) 
        out = np.divide(np.square(np.std(layer_being_process,0)),np.mean(np.square(layer_being_process),0))
        out = np.sort(out)
        print(out[int(len(out)/2)])
        """
        plt.hist(out[~np.isnan(out)],bins=10)
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        """

def pca_calculation(layer):
        # Setup sklearn library for PCA analysis
        pca = PCA(n_components=5)
        # Calculate Explained Variance
        layer_being_process = layer
        layer_being_process = np.reshape(layer_being_process, (len(layer_being_process),-1)) 
        pca.fit(layer_being_process)
        print(pca.explained_variance_ratio_)

#Run CNN
def test_CNN(x):

    # prediction = model(x)
    prediction, conv1, conv1_pool, conv2, conv2_pool, fc= model(x) # To output layer by layer info
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

        # Container for layer outputs
        conv1_out = []
        conv1_pool_out = []
        conv2_out = []
        conv2_pool_out = []
        fully_connected = []

        # Run Batches
        for _ in range(batch_count):
            epoch_x,epoch_y = data.next_img_batch(batch_size)
            correct = tf.equal(tf.arg_max(prediction,1), y)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # Output Layer Info
            acc,  pd, c1, cp1, c2, cp2, f_connect = sess.run([accuracy, prediction, conv1,conv1_pool,conv2,conv2_pool,fc], feed_dict = {x: epoch_x, y: epoch_y})
            conv1_out.extend(c1)
            conv1_pool_out.extend(cp1)
            conv2_out.extend(c2)
            conv2_pool_out.extend(cp2)
            fully_connected.extend(f_connect)
            temporary_accuracy = temporary_accuracy + acc

        # Add up accuracy
        test_accuracy = (temporary_accuracy/batch_count)
        print("Test Accuracy is ", test_accuracy)
        plot_hist_of_layer(conv1_out)
        plot_hist_of_layer(conv1_pool_out)
        plot_hist_of_layer(conv2_out)
        plot_hist_of_layer(conv2_pool_out)
        plot_hist_of_layer(fully_connected)
        plot_hist_of_layer(pd)
        # pca_calculation(data.images)

# call main
if __name__ == '__main__':
    test_CNN(x)