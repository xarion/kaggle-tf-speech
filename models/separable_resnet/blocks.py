import tensorflow as tf
from tensorflow.contrib import layers


class Blocks:
    def __init__(self, training=True, parameters=None):
        self.parameters = parameters
        self.filter_size = self.parameters["filter_size"]
        self.stride_length = self.parameters["stride_length"]
        self.training = training
        self.decayed_variables = []
        self.total_parameters = 0l

    def weight_variable(self, name, shape):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=layers.xavier_initializer())

    def weight_variable_with_initializer(self, name, shape, initializer):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer)

    def relu(self, input_layer):
        relu = tf.nn.relu(input_layer)
        return relu

    def conv2d(self, input_layer, filter_size, input_channels, output_channels, strides):
        weights = self.weight_variable("conv_weights", [filter_size, 1, input_channels, output_channels])
        self.decayed_variables.append(weights)
        logits = tf.nn.conv2d(input_layer, weights, strides=[1, strides, 1, 1], padding="SAME")
        logits = self.add_bias(logits, output_channels)
        return logits

    def separable_conv2d(self, input_layer, filter_size, input_channels, depthwise_multiplier, output_channels,
                         strides, depthwise_relu_bn=True):
        intermediate_channels = input_channels * depthwise_multiplier

        with tf.variable_scope('depthwise'):
            depthwise_weights = self.weight_variable("depthwise_weights",
                                                     [filter_size, 1, input_channels, depthwise_multiplier])
            self.decayed_variables.append(depthwise_weights)
            depthwise_results = tf.nn.depthwise_conv2d_native(input=input_layer,
                                                              filter=depthwise_weights,
                                                              strides=[1, strides, 1, 1],
                                                              padding="SAME",
                                                              name="depthwise")
            depthwise_results = self.add_bias(depthwise_results, intermediate_channels)

            if depthwise_relu_bn:
                depthwise_results = self.batch_normalization(depthwise_results)
                depthwise_results = self.relu(depthwise_results)

        with tf.variable_scope('pointwise'):
            pointwise_weights = self.weight_variable("pointwise_weights",
                                                     [1, 1, intermediate_channels, output_channels])
            self.decayed_variables.append(pointwise_weights)
            pointwise_results = tf.nn.conv2d(depthwise_results,
                                             pointwise_weights,
                                             [1, 1, 1, 1],
                                             padding="SAME")

            pointwise_results = self.add_bias(pointwise_results, output_channels)

        return pointwise_results

    def residual_separable(self, input_layer, input_channels, output_channels,
                           strides, activate_before_residual):

        if activate_before_residual:
            input_layer = self.batch_normalization(input_layer)
            input_layer = self.relu(input_layer)
            residual = input_layer
        else:
            residual = input_layer
            input_layer = self.batch_normalization(input_layer)
            input_layer = self.relu(input_layer)

        with tf.variable_scope("convolution_1"):
            features = self.separable_conv2d(input_layer,
                                             filter_size=9,
                                             input_channels=input_channels,
                                             depthwise_multiplier=1,
                                             output_channels=output_channels,
                                             strides=strides)

        with tf.variable_scope("convolution_2"):
            features = self.batch_normalization(features)
            features = self.relu(features)
            features = self.separable_conv2d(features,
                                             filter_size=9,
                                             input_channels=output_channels,
                                             depthwise_multiplier=1,
                                             output_channels=output_channels,
                                             strides=1)

        with tf.variable_scope("residual_connection"):
            if strides is not 1 or input_channels != output_channels:
                residual = tf.nn.max_pool(residual,
                                          ksize=[1, strides, 1, 1],
                                          strides=[1, strides, 1, 1],
                                          padding="SAME")

                residual = self.pad_residual_features(residual,
                                                      input_channels,
                                                      output_channels)
            connection = self.residual_connection(residual, features)

        return connection

    def residual_conv2d(self, input_layer, input_channels, output_channels,
                        strides, activate_before_residual):

        if activate_before_residual:
            input_layer = self.batch_normalization(input_layer)
            input_layer = self.relu(input_layer)
            residual = input_layer
        else:
            residual = input_layer
            input_layer = self.batch_normalization(input_layer)
            input_layer = self.relu(input_layer)

        with tf.variable_scope("convolution_1"):
            features = self.conv2d(input_layer,
                                   filter_size=self.filter_size,
                                   input_channels=input_channels,
                                   output_channels=output_channels,
                                   strides=strides)

        with tf.variable_scope("convolution_2"):
            features = self.batch_normalization(features)
            features = self.relu(features)
            features = self.conv2d(features,
                                   filter_size=self.filter_size,
                                   input_channels=output_channels,
                                   output_channels=output_channels,
                                   strides=1)

        with tf.variable_scope("residual_connection"):
            if strides is not 1 or input_channels != output_channels:
                residual = tf.nn.max_pool(residual,
                                          ksize=[1, strides, 1, 1],
                                          strides=[1, strides, 1, 1],
                                          padding="SAME")

                residual = self.pad_residual_features(residual,
                                                      input_channels,
                                                      output_channels)
            connection = self.residual_connection(residual, features)

        return connection

    def pad_residual_features(self, residual, input_channels, output_channels):
        half_channel_difference = (output_channels - input_channels) // 2

        beginning_pad_count = tf.Variable(initial_value=half_channel_difference,
                                          name="beginning_pad_count")
        ending_pad_count = tf.Variable(initial_value=output_channels - (input_channels + half_channel_difference),
                                       name="ending_pad_count")
        residual = tf.pad(residual,
                          [[0, 0], [0, 0], [0, 0], [beginning_pad_count, ending_pad_count]])
        return residual

    def residual_connection(self, residual, current):
        return tf.add(residual, current)

    def batch_normalization(self, input_layer):
        bn = layers.batch_norm(input_layer, fused=True, trainable=self.training, is_training=self.training, scale=True)
        return bn

    def add_bias(self, layer, number_of_channels):
        bias = self.weight_variable_with_initializer("bias", [number_of_channels], tf.zeros_initializer())
        return tf.nn.bias_add(layer, bias)

    def fc(self, input_layer, input_channels, output_channels):
        weights = self.weight_variable("fc_weights", [input_channels, output_channels])
        self.decayed_variables.append(weights)
        return tf.matmul(input_layer, weights)

    def normalized_fc(self, input_layer, input_channels, output_channels):
        fc = self.biased_fc(input_layer, input_channels, output_channels)
        fc = tf.expand_dims(fc, 1)
        fc = tf.expand_dims(fc, 1)
        bn = self.batch_normalization(fc)
        return tf.squeeze(bn, axis=[1, 2])

    def biased_fc(self, input_layer, input_channels, output_channels):
        pre_bias = self.fc(input_layer, input_channels, output_channels)
        return self.add_bias(pre_bias, output_channels)

    def deconvolution(self, input_layer, filter_size, input_channels, output_channels, output_dimensions, strides):
        weights = self.weight_variable("deconv_weights", [filter_size, 1, output_channels, input_channels])
        self.decayed_variables.append(weights)
        output_shape = [output_dimensions[0],
                        output_dimensions[1],
                        output_dimensions[2],
                        weights.get_shape()[2].value]
        deconv = tf.nn.conv2d_transpose(input_layer,
                                        weights,
                                        output_shape,
                                        strides=[1, strides, 1, 1],
                                        padding="SAME")

        return deconv

    def get_decayed_variables(self):
        return self.decayed_variables

    def reduce_var(self, x, axis=None, keepdims=False):
        m = tf.reduce_mean(x, axis=axis, keep_dims=True)
        devs_squared = tf.square(x - m)
        return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)
