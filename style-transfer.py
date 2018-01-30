"""This file implements style-transfer by Gatys et al."""

import tensorflow as tf
import tensorflow.contrib.layers as l
import numpy as np
import scipy.misc as sc
from PIL import Image
from vgg import Vgg19, vggnet

img_size = 224


# deep net settings
layer_sizes = [32, 64, 128, 128]
kernels = [5, 5, 5, 5]
n = len(layer_sizes)


NET = vggnet("./vgg19.npy", [3, 4])
N_LAYERS = 2
# ## multi-scale settings
n_channels_per_scale = 20
# use kernels in power of 2 and l kernels of each to get enough info
kernels = [33,17,9,5]
nkernels = len(kernels)


def msnet(x):
    """Implement a shallow multi-scale convnet to capture style of an image"""

    features = tf.stack([tf.reshape(
        l.avg_pool2d(
            l.conv2d(x, n_channels_per_scale,
                          padding = "same",
                          kernel_size=kernels[i],
                          activation_fn=tf.nn.leaky_relu,
                          trainable=False,
                          biases_initializer=tf.random_uniform_initializer(maxval=0.1),
                          reuse=tf.AUTO_REUSE,
                          scope='style%d' % i),
            2),  # pool op
        (3, -1, n_channels_per_scale))  # reshape op
        for i in range(nkernels)]) # stack op
    
    features=tf.transpose(features, (1,2,3,0))

    return tf.reshape(features, (3, -1, n_channels_per_scale*nkernels))

def net(x):
    """ Neural net for style and content representations
    --  conv l with channels as specified in l with leaky ReLU activations
    """

    outs = []
    out = x
    for i, nc in enumerate(layer_sizes):
        out = l.conv2d(out, nc, kernel_size=kernels[i],
                            padding="same",
                            activation_fn=tf.nn.leaky_relu,
                            trainable=False,
                            reuse=tf.AUTO_REUSE,
                            biases_initializer=tf.random_uniform_initializer(
                                maxval=1/sum(layer_sizes[i:i+2])),
                            scope='conv%d' % i)
        out = l.avg_pool2d(out, 2)
        outs.append(tf.reshape(out, (3, -1, nc)))

    return outs


def getgram(image_feature, axis=0):
    return tf.tensordot(image_feature, image_feature, axes=[[axis], [axis]])


def merge(image, content, style, loss_at_level, nn):

    with tf.name_scope('apply_nn'):
        outputs = nn(tf.stack((content, style, image)))

    
    with tf.name_scope('merge'):
        content_loss = style_loss = 0

        if nn.__name__!='msnet':

            content_loss = tf.nn.l2_loss(outputs[0][0] - outputs[0][2])
            # use for loop
            for i in range(N_LAYERS):
                print(outputs[i].get_shape().as_list())
                outc = outputs[i][0]
                outs = outputs[i][1]
                outx = outputs[i][2]
                N1, N2 = outc.get_shape().as_list()

                # content_loss += tf.nn.l2_loss(outc-outx)
                Gs = getgram(outs)
                Gx = getgram(outx)
                style_loss += tf.nn.l2_loss(Gs-Gx) / (2*N1*N1*N2*N2)

        else:
            outc, outs, outx = tf.split(axis=0, num_or_size_splits=3, value=outputs)
            _, N1, N2 = outc.get_shape().as_list() # gets size 1 axis 0
            content_loss += tf.nn.l2_loss(outc-outx)
            Gs = getgram(outs)
            Gx = getgram(outx)
            style_loss += tf.nn.l2_loss(Gs-Gx) / (2*N1*N1*N2*N2)

    return content_loss, style_loss, outputs


def train(loss):
    return tf.train.AdamOptimizer(learning_rate=5e-1).minimize(loss)


def visualize(out, desc='content'):
    if type(out) == list:
        nk, nf, nc = len(out), len(out[0]), len(out[0][0])
    else:
        nk, nf, nc = out.shape  # nkernels, nfeatures, channels_per_kernel

    for k in range(nk):
        outk = out[k]
        # figure out how big a canvas is needed for the feature
        s = len(outk[:, 0])
        s = int(np.sqrt(s))
        ncols = 5
        nrows = int(np.ceil(nc/float(ncols))) + 1

        total_height = s*ncols + (ncols-1)*5  # 5 px gap between two outputs
        total_width = s*nrows + (nrows-1)*5  # 5 px gap between two outputs

        new_im = Image.new('P', (total_height, total_width))

        for i in range(nc):
            c = i % ncols
            r = i//ncols
            x_offset = c*s+5
            y_offset = r*s+5
            offset = (x_offset, y_offset)
            g = outk[:, i].reshape(s, s)
            g = (g-g.min())/(g.max()-g.min())*255
            im = Image.fromarray(np.uint8(g))
            new_im.paste(im, box=offset)

        new_im.save('activation_%s_kernel%d.png' % (desc, k))


def activations(sess, ops, feed_dict, labels):
    with tf.name_scope('visualize'):
        results = sess.run(ops, feed_dict=feed_dict)

    for i, result in enumerate(results):
        visualize(result, desc=labels[i])


def style_transfer(c, s, nn=vggnet, mix=(10, 100, 1)):
    """transfer style of s onto c"""
    h, w, nc = c.shape

    tf.reset_default_graph()

    content = tf.placeholder('float32', shape=(h, w, nc))
    style = tf.placeholder('float32', shape=(h, w, nc))
    x = tf.Variable(tf.random_uniform((h, w, nc)), name='input')

    lc, ls, outs = merge(x, content, style, -1, nn)

    # content + perceptual + tv loss
    loss = mix[0]*lc + mix[1]*ls + mix[2]*tf.image.total_variation(x)
    train_step = train(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with tf.name_scope('optimize'):

            # run optimization
            for ii in range(50001):
                closs, sloss, tloss, _, _ = sess.run(
                    [lc, ls, loss, outs, train_step],
                    feed_dict={content: c, style: s})
                if ii % 1000 == 0 or tloss < 1e-6:

                    print("loss @ iteration %d = %f, in content = %f, in style = %f" %
                          (ii+1, tloss, closs, sloss))

                    out = sess.run(x)
                    sc.imsave('output.jpg', out.reshape(h, w, nc))
                    if tloss < 1e-6:
                        break

    return out


def main():
    content = sc.imresize(sc.imread('tubingen.jpg'),
                          (img_size, img_size))/255.0
    sc.imsave('content.jpg', content)

    # # corruption
    # p = 0.1
    # npixels = img_size**2
    # pixel_index = np.random.randint(0, npixels*3, int(p*npixels*3))
    # channel = pixel_index//npixels
    # row = (pixel_index - channel*npixels)//img_size
    # col = (pixel_index - channel*npixels)%img_size
    # print(content[row, col, channel].shape)
    # content[row, col, channel] = 0.0

    # # save corrupted image
    # sc.imsave('corrupted_content.jpg', content)

    style = sc.imresize(sc.imread('starry_night.jpg'),
                        (img_size, img_size))/255.0
    # style = sc.imresize(sc.imread('perforated_0044.jpg'),
    #                     (img_size, img_size))/255.0
    sc.imsave('style.jpg', style)

    out = style_transfer(content, style, nn=NET, mix=(7.5, 1000, 0.001))
    return

if __name__ == '__main__':
    main()
