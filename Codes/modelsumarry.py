import numpy as np

from tensorflow.python.layers import base
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, conv2d_transpose
import tensorflow.contrib.slim as slim

def sumarr(arr):
    S = 1
    new_arr = np.uint16(arr)
    for i in new_arr:
        S *= i
    return S

#x = np.zeros((1,4,4,3))
x = np.zeros((1, 128, 128, 12))
print ("x shape: ", x.shape)
x_tf = tf.convert_to_tensor(x, np.float32)

print ("x_tf shape: ", x_tf.shape)
#z_tf = tf.layers.conv2d(x_tf, filters=32, kernel_size=(3,3))

features_root=64
filter_size=3
pool_size=2
output_channel=1
layers = 4

features_size = 0

in_node = x_tf
print ("in_node shape: ", in_node.shape)

conv = []
for layer in range(0, layers):
    features = 2**layer*features_root

    conv1 = conv2d(inputs=in_node, num_outputs=features, kernel_size=filter_size)
    features_size += sumarr(conv1.shape)
    print ("conv1 shape:", conv1.shape)

    conv2 = conv2d(inputs=conv1, num_outputs=features, kernel_size=filter_size)
    features_size += sumarr(conv2.shape)
    print ("conv2 shape:", conv2.shape)

    conv.append(conv2)

    if layer < layers - 1:
        in_node = max_pool2d(inputs=conv2, kernel_size=pool_size, padding='SAME')
        #in_node = conv2d(inputs=conv2, num_outputs=features, kernel_size=filter_size, stride=2)

in_node = conv[-1]

for layer in range(layers-2, -1, -1):
    features = 2**(layer+1)*features_root

    h_deconv = conv2d_transpose(inputs=in_node, num_outputs=features//2, kernel_size=pool_size, stride=pool_size)
    features_size += sumarr(h_deconv.shape)

    h_deconv_concat = tf.concat([conv[layer], h_deconv], axis=3)
    features_size += sumarr(h_deconv_concat.shape)

    conv1 = conv2d(inputs=h_deconv_concat, num_outputs=features//2, kernel_size=filter_size)
    features_size += sumarr(conv1.shape)

    in_node = conv2d(inputs=conv1, num_outputs=features//2, kernel_size=filter_size)
    features_size += sumarr(conv2.shape)

output = conv2d(inputs=in_node, num_outputs=output_channel, kernel_size=filter_size, activation_fn=None)

z_tf = tf.tanh(output)

def model_summary():
    model_vars = tf.trainable_variables()
    name = [v.name for v in model_vars]
    print ("name_model_vars:", name)
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()
print ("features_size:", features_size)
print ("features byte:", features_size*4)
#tf.summary.get_summary_description(z_tf)
