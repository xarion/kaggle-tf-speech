import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, batch_norm

MAX_POOLING = 1


class Blocks:
    def __init__(self):
        self.weights = []

    def create_weights(self, name, shape):
        w = tf.get_variable(name, shape=shape, initializer=xavier_initializer())
        self.weights.append(w)
        return w

    def create_weights_with_initializer(self, name, shape, initializer):
        w = tf.get_variable(name, shape=shape, initializer=initializer)
        self.weights.append(w)
        return w

    def conv2d(self, input_layer, filter_shape, stride=1, dilation=1):
        with tf.variable_scope("conv2d"):
            _filter = self.create_weights("filter_weights", filter_shape)
            return tf.nn.convolution(input_layer, _filter, strides=[stride, stride], padding="SAME",
                                     dilation_rate=[dilation, dilation])

    def relu_conv2d(self, input_layer, filter_shape, stride=1, mcrelu=False):
        with tf.variable_scope("relu_conv2d"):
            l = self.conv2d(input_layer, filter_shape, stride=stride)
            return self.normalized_relu_activation(l, negative_concatenation=mcrelu)

    def normalized_relu_activation(self, input_layer, negative_concatenation=False):
        if negative_concatenation:
            input_layer = tf.concat([input_layer, -1 * input_layer], 3)

        input_layer = self.batch_normalization(input_layer)

        return tf.nn.relu(input_layer)

    @staticmethod
    def batch_normalization(input_layer):
        with tf.variable_scope("batch_norm"):
            bn = batch_norm(input_layer, fused=True, scale=True)
            return bn

    def block(self, layer, down_sampling, kernel_sizes, strides, dilation_rates, input_channel, output_channels):

        layer = tf.nn.max_pool(layer, ksize=[1, 8, 1, 1], strides=[1, 4, 1, 1], padding="SAME")

        if input_channel != output_channels[-1]:
            layer = self.normalized_relu_activation(layer)
            with tf.variable_scope("identity_mapping"):
                residual = self.conv2d(layer,
                                       filter_shape=[1, 1, input_channel, output_channels[-1]],
                                       stride=1)
        else:
            residual = layer
            layer = self.normalized_relu_activation(layer)

        for c in range(0, len(kernel_sizes)):
            with tf.variable_scope("inner_convolution_%d" % c):
                if c is not 0:
                    layer = self.normalized_relu_activation(layer)
                layer = self.conv2d(layer, [kernel_sizes[c], kernel_sizes[c], input_channel, output_channels[c]],
                                    stride=strides[c], dilation=dilation_rates[c])
                input_channel = output_channels[c]

        layer = layer + residual
        return layer
