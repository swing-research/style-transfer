"""This file implements style-transfer by Gatys et al."""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import scipy.misc as sms
import scipy.misc as sc
from PIL import Image


layer_sizes = [32]
kernel_size = [20]
n = len(layer_sizes)
img_size = 256


def msnet(x):
    """Implement a shallow multi-scale convnet to capture style of an image"""
    l = 10  # number of channels per scale
    # use kernels in power of 2 and l kernels of each to get enough info
    nkernels = int(np.log(img_size)/np.log(2))
    kernels = [img_size//2**i for i in range(1, nkernels+1)]

    inp = tf.reshape(x, (1, img_size, img_size, 3))
    style_features = tf.stack([tf.reshape(
        layers.avg_pool2d(
            layers.conv2d((inp), l,
                          kernel_size=kernels[i],
                          activation_fn=tf.nn.leaky_relu,
                          trainable=False,
                          reuse=tf.AUTO_REUSE,
                          scope='style%d' % i), 2), (-1, l))
        for i in range(nkernels)])
    return style_features


def net(x):
    """ Neural net for style and content representations
    --  conv layers with channels as specified in layerswith leaky ReLU activations
    """

    kernels = kernel_size

    if type(kernel_size) == int:
        kernels = [kernel_size, ]*n

    elif type(kernel_size) == list:
        if len(kernel_size) < n:
            kernels += [kernel_size[-1], ]*(n-len(kernel_size))

    else:
        kernels = []

    outs = []
    # out = tf.clip_by_value(tf.reshape(x,(1,img_size,img_size,3)),0,1)
    out = tf.reshape(x, (1, img_size, img_size, 3))
    for i, nc in enumerate(layer_sizes):
        out = layers.conv2d(out, nc, kernel_size=kernels[i],
                            activation_fn=tf.nn.leaky_relu,
                            trainable=False,
                            reuse=tf.AUTO_REUSE,
                            scope='conv%d' % i)
        out = layers.avg_pool2d(out, 2)
        outs.append(tf.reshape(out, (-1, nc)))

    return outs


def getgram(image_feature):
    return tf.tensordot(image_feature, image_feature, axes=[[0], [0]])


def merge(image, content, style, loss_at_level, weights=0.0):

    if type(weights) != list:
        wts = [1.0/n, ]*n
    else:
        if len(weights) != n:
            wts = [1.0/n, ]*n
        else:
            wts = weights

    with tf.name_scope('merge'):
        outc = msnet(content)
        outx = msnet(image)
        outs = msnet(style)

        if loss_at_level == -1:
            content_loss = 0
            style_loss = 0
            # use loss from all levels
            for i in range(n):
                content_loss += wts[i] * \
                    tf.reduce_sum((outc[i] - outx[i])**2)*0.5
                g = tf.cast(tf.shape(outs[i]), 'float32')
                style_loss += wts[i]*tf.reduce_sum(
                    (getgram(outs[i]) - getgram(outx[i]))**2)*0.25/g[1]/g[1]
        else:
            content_loss = tf.reduce_sum(
                outc[loss_at_level] - outx[loss_at_level])**2*0.5
            g = tf.cast(tf.shape(outs[loss_at_level]), 'float32')
            style_loss = tf.reduce_sum(
                (getgram(outs[loss_at_level]) - getgram(outx[loss_at_level]))**2)*0.25/g[1]/g[1]

    return content_loss, style_loss


def train(im, content, style, a=0.2, loss_at_level=-1):

    content_loss, style_loss = merge(im, content, style, loss_at_level)
    loss = a*content_loss + (1-a)*style_loss
    with tf.name_scope('optimize'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    return train_step,  loss, content_loss, style_loss


def visualize(out, desc='content'):
    last_layer = out[-1]
    s, nc = last_layer.shape
    ncols = 8
    nrows = int(np.ceil(nc/float(ncols))) + 1

    s = int(np.sqrt(s))
    total_width = s*ncols + (ncols-1)*5
    total_height = s*nrows + (nrows-1)*5

    new_im = Image.new('P', (total_width, total_height))
    for i in range(nc):
        c = i % ncols
        r = i//ncols
        x_offset = c*s+5
        y_offset = r*s+5
        g = last_layer[:, i].reshape(s, s)
        g = (g-g.min())/(g.max()-g.min())*255
        im = Image.fromarray(np.uint8(g))
        new_im.paste(im, box=(x_offset, y_offset+s))

    new_im.save('outputs_%s.png' % desc)
    return


def style_transfer(c, s):
    """transfer style of s onto c"""
    h, w, nc = c.shape

    tf.reset_default_graph()

    content = tf.placeholder('float32')
    style = tf.placeholder('float32')
    x = tf.Variable(tf.random_uniform((1, h, w, nc)), name='input')

    op_train, op_loss, op_content, op_style = train(
        x, content, style, a=0.5, loss_at_level=-1)
    op_c, op_s = net(content), net(style)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # with tf.name_scope('visualize'):
        #     outsc, outss = sess.run([op_c, op_s], feed_dict={
        #         content: c, style: s})
        #     visualize(outsc, desc='content')
        #     visualize(outss, desc='style')

        with tf.name_scope('optimize'):

            for ii in range(20001):
                _, loss, closs, sloss = sess.run(
                    [op_train, op_loss, op_content, op_style],
                    feed_dict={content: c, style: s})
                if ii % 1000 == 0 or loss < 1e-6:
                    print("loss @ iteration %d = %f, in content = %f, in style = %f" %
                          (ii+1, loss, closs, sloss))
                    out = sess.run(x)
                    sc.imsave('output.jpg', out.reshape(img_size, img_size, 3))
                    if loss < 1e-6:
                        break

    return out


def main():
    content = sc.imresize(sc.imread('tubingen.jpg'),
                          (img_size, img_size))/255.0
    style = sc.imresize(sc.imread('starry_night.jpg'),
                        (img_size, img_size))/255.0
    # sc.imsave('content.jpg', content)
    # sc.imsave('style.jpg', style)
    out = style_transfer(content, style)
    return

if __name__ == '__main__':
    main()
