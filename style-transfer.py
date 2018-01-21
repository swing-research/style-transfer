"""This file implements style-transfer by Gatys et al."""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import scipy.misc as sc
from PIL import Image

img_size = 256

## deep net settings
layer_sizes = [32,64,128,3]
kernel_size = [32,16,16,4]
n = len(layer_sizes)

## multi-scale settings
n_channels_per_scale = 100
# use kernels in power of 2 and l kernels of each to get enough info
kernels = [img_size//2**i for i in range(5, int(np.log(img_size)/np.log(2)))]
nkernels = len(kernels)


def msnet(x):
    """Implement a shallow multi-scale convnet to capture style of an image"""

    inp = tf.reshape(x, (1, img_size, img_size, 3))
    style_features = tf.stack([tf.reshape(
        layers.avg_pool2d(
            layers.conv2d((inp), n_channels_per_scale,
                          padding = "valid",
                          kernel_size=kernels[i],
                          activation_fn=tf.nn.leaky_relu,
                          trainable=False,
                          reuse=tf.AUTO_REUSE,
                          scope='style%d' % i),
            4),  # pool op
        (-1, n_channels_per_scale))  # reshape op
        for i in range(nkernels)]) # stack op
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
                            padding="valid",
                            activation_fn=tf.nn.leaky_relu,
                            trainable=False,
                            reuse=tf.AUTO_REUSE,
                            scope='conv%d' % i)
        out = layers.avg_pool2d(out, 4)
        outs.append(tf.reshape(out, (-1, nc) ))

    return outs


def getgram(image_feature):
    return tf.tensordot(image_feature, image_feature, axes=[[1], [1]])


def merge(image, content, style, loss_at_level, nn):

    n = len(kernels)

    with tf.name_scope('apply_nn'):
        outc = nn(content)
        outx = nn(image)
        outs = nn(style)
    
    with tf.name_scope('merge'):

        if loss_at_level == "all":
            if type(outs) == list:
                content_loss = style_loss = 0

                # use for loop
                for i in range(len(outc)):
                    content_loss += tf.reduce_sum((outc[i]-outx[i])**2)*0.5
                    style_loss += tf.reduce_sum((getgram(outs[i]) - getgram(outx[i]))**2)*0.25/tf.reduce_sum(getgram(outs[i])**2)

            else:
                # if not list, use tensor ops

                # use loss from all levels/ all scales
                content_loss = tf.reduce_sum((outc - outx)**2)*0.5
                style_loss =  tf.reduce_sum((getgram(outs) - getgram(outx))**2)*0.25/tf.reduce_sum(getgram(outs)**2)

        else:
            content_loss = tf.reduce_sum(
                outc[loss_at_level] - outx[loss_at_level])**2*0.5
            g = tf.cast(tf.shape(outs[loss_at_level]), 'float32')
            style_loss = tf.reduce_sum(
                (getgram(outs[loss_at_level]) - getgram(outx[loss_at_level]))**2)*0.25/(g[1]**2)/(g[0]**2)

    return content_loss, style_loss


def train(im, content, style, nn, a=0.0, loss_at_level="all"):

    content_loss, style_loss = merge(im, content, style, loss_at_level, nn)
    
    with tf.name_scope('optimize'):
        loss = a*content_loss + (1-a)*style_loss*1e3
        # loss = tf.maximum(content_loss, style_loss*1e3, name="loss")
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    return train_step,  loss, content_loss, style_loss

def visualize(out, desc='content'):
    if type(out) == list:
        nk, nf, nc = len(out), len(out[0]), len(out[0][0])
    else:
        nk, nf, nc = out.shape # nkernels, nfeatures, channels_per_kernel

    for k in range(nk):
        outk = out[k]
        # figure out how big a canvas is needed for the feature
        s = len(outk[:,0])
        s = int(np.sqrt(s))
        ncols = 5
        nrows = int(np.ceil(nc/float(ncols))) + 1

        total_height = s*ncols + (ncols-1)*5 # 5 px gap between two outputs
        total_width = s*nrows + (nrows-1)*5 # 5 px gap between two outputs

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

        new_im.save('activation_%s_kernel%d.png' % (desc, kernels[k]) )



# def visualize(out, desc='content'):
#     last_layer = out[-1]
#     s, nc = last_layer.shape
#     ncols = 8
#     nrows = int(np.ceil(nc/float(ncols))) + 1

#     s = int(np.sqrt(s))
#     # height and width are opposites in Image lib
#     total_height = s*ncols + (ncols-1)*5
#     total_width = s*nrows + (nrows-1)*5

#     new_im = Image.new('P', (total_width, total_height))
#     for i in range(nc):
#         c = i % ncols
#         r = i//ncols
#         x_offset = c*s+5
#         y_offset = r*s+5
#         g = last_layer[:, i].reshape(s, s)
#         g = (g-g.min())/(g.max()-g.min())*255
#         im = Image.fromarray(np.uint8(g))
#         new_im.paste(im, box=(x_offset, y_offset+s))

#     new_im.save('corrupted_%s.png' % desc)
#     return


def activations(sess, ops, feed_dict, labels):
    with tf.name_scope('visualize'):
        results = sess.run(ops, feed_dict=feed_dict)

    for i, result in enumerate(results):
        visualize(result, desc=labels[i])


def style_transfer(c, s, nn=msnet, mix=0.5):
    """transfer style of s onto c"""
    h, w, nc = c.shape

    tf.reset_default_graph()

    content = tf.placeholder('float32')
    style = tf.placeholder('float32')
    x = tf.Variable(tf.random_uniform((1, h, w, nc)), name='input')

    op_train, op_loss, op_content, op_style = train(
        x, content, style, nn, a=mix, loss_at_level="all")
    op_c, op_s, op_x = nn(content), nn(style), nn(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with tf.name_scope('optimize'):
            # visualize output of our content and style filters
            activations(sess, [op_c, op_s], {content: c, style: s}, ['content', 'style'])

            # run optimization
            for ii in range(50001):
                _, loss, closs, sloss = sess.run(
                    [op_train, op_loss, op_content, op_style],
                    feed_dict={content: c, style: s})
                if ii % 1000 == 0 or loss < 1e-6:
                    activations(sess, [op_x], {}, ['variable'])
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
    sc.imsave('content.jpg', content)

    # corruption
    p = 0.1
    npixels = img_size**2
    pixel_index = np.random.randint(0, npixels*3, int(p*npixels*3))
    channel = pixel_index//npixels
    row = (pixel_index - channel*npixels)//img_size
    col = (pixel_index - channel*npixels)%img_size
    print(content[row, col, channel].shape)
    content[row, col, channel] = 0.0

    # save corrupted image
    sc.imsave('corrupted_content.jpg', content)


    style = sc.imresize(sc.imread('starry_night.jpg'),
                        (img_size, img_size))/255.0
    # style = sc.imresize(sc.imread('perforated_0044.jpg'),
    #                     (img_size, img_size))/255.0
    sc.imsave('style.jpg', style)

    out = style_transfer(content, style, nn=net, mix=0.5)
    return

if __name__ == '__main__':
    main()
