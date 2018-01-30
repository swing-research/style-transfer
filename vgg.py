import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:

    def __init__(self, vgg19_npy_path=None, layers=[]):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.layers = layers
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(
            axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.data_dict = None

        self.ops = []

        for val in self.layers:
            if val == 1:
                a, b, c, d = self.conv1_2.get_shape().as_list()
                self.ops.append(tf.reshape(self.conv1_2, (a, b*c, d)))
            elif val == 2:
                a, b, c, d = self.conv2_2.get_shape().as_list()
                self.ops.append(tf.reshape(self.conv2_2, (a, b*c, d)))
            elif val == 3:
                a, b, c, d = self.conv3_4.get_shape().as_list()
                self.ops.append(tf.reshape(self.conv3_4, (a, b*c, d)))
            elif val == 4:
                a, b, c, d = self.conv4_4.get_shape().as_list()
                self.ops.append(tf.reshape(self.conv4_4, (a, b*c, d)))
            elif val == 5:
                a, b, c, d = self.conv5_4.get_shape().as_list()
                self.ops.append(tf.reshape(self.conv5_4, (a, b*c, d)))

        print(("build model finished: %ds" % (time.time() - start_time)))

        return self.ops

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def vggnet(path, layers):
    vgg19 = randomVgg19(vgg19_npy_path=path, layers=layers)
    return vgg19.build


class randVgg19(Vgg19):

    def __init__(self, path="./vgg19.npy", layers=[]):
        super.__init__(path, layers)
        self.sizes = [64, 128, 256, 512, 512]
        self.layers = layers

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            blockid = int(name[-3])
            out = tf.contrib.layers.conv2d(bottom, self.sizes[block],
                                           kernel_size=kernels[i],
                                           padding="same",
                                           activation_fn=tf.nn.relu,
                                           trainable=False,
                                           reuse=tf.AUTO_REUSE,
                                           biases_initializer=tf.random_uniform_initializer(
                maxval=0.1),
                scope='conv%d' % i)

            return out
