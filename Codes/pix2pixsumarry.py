import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base

import collections
import functools

def sumarr(arr):
    S = 1
    new_arr = np.uint16(arr)
    for i in new_arr:
        S *= i
    return S



layers = tf.contrib.layers

import tensorflow.contrib.slim as slim

features_size = 0
def pix2pix_discriminator(net, num_filters, padding=2, is_training=False):
    """Creates the Image2Image Translation Discriminator.
    Args:
      net: A `Tensor` of size [batch_size, height, width, channels] representing
        the input.
      num_filters: A list of the filters in the discriminator. The length of the
        list determines the number of layers in the discriminator.
      padding: Amount of reflection padding applied before each convolution.
      is_training: Whether or not the model is training or testing.
    Returns:
      A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
      'patches' we're attempting to discriminate and a dictionary of model end
      points.
    """
    global features_size
    del is_training
    end_points = {}

    num_layers = len(num_filters)

    def padded(net, scope):
        if padding:
            with tf.variable_scope(scope):
                spatial_pad = tf.constant(
                    [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                    dtype=tf.int32)
                return tf.pad(net, spatial_pad, 'REFLECT')
        else:
            return net

    with tf.contrib.framework.arg_scope(
            [layers.conv2d],
            kernel_size=[4, 4],
            stride=2,
            padding='valid',
            activation_fn=tf.nn.leaky_relu):

        # No normalization on the input layer.
        net = layers.conv2d(
            padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')
        features_size += sumarr(net.shape)

        end_points['conv0'] = net

        for i in range(1, num_layers - 1):
            net = layers.conv2d(
                padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)
            end_points['conv%d' % i] = net
            features_size += sumarr(net.shape)

        # Stride 1 on the last layer.
        net = layers.conv2d(
            padded(net, 'conv%d' % (num_layers - 1)),
            num_filters[-1],
            stride=1,
            scope='conv%d' % (num_layers - 1))
        end_points['conv%d' % (num_layers - 1)] = net
        features_size += sumarr(net.shape)

        # 1-dim logits, stride 1, no activation, no normalization.
        logits = layers.conv2d(
            padded(net, 'conv%d' % num_layers),
            1,
            stride=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv%d' % num_layers)
        end_points['logits'] = logits
        features_size += sumarr(net.shape)
        end_points['predictions'] = tf.sigmoid(logits)
    return logits, end_points

x = np.zeros((1, 128, 128, 12))
print ("x shape: ", x.shape)
x_tf = tf.convert_to_tensor(x, np.float32)

logits, end_points = pix2pix_discriminator(x_tf, num_filters=(128, 256, 512, 512))

z_tf = logits
#z_tf.summary()

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()
print ("features_size:", features_size)
print ("features byte:", features_size*4)
