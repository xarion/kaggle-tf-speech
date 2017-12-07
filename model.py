import tensorflow as tf

from blocks import Blocks


class Model:
    def __init__(self, data):
        self.input = data.inputs
        self.labels = data.labels
        self.blocks = Blocks()

        with tf.variable_scope("conv_1"):
            conv_1 = self.blocks.conv2d(self.input, [8, 20, 1, 64])
            conv_1 = self.blocks.normalized_relu_activation(conv_1)
            conv_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("conv_2"):
            conv_2 = self.blocks.conv2d(conv_1, [4, 10, 64, 128])
            conv_2 = self.blocks.normalized_relu_activation(conv_2)
            conv_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("conv_3"):
            conv_3 = self.blocks.conv2d(conv_2, [2, 5, 128, 256])
            conv_3 = self.blocks.normalized_relu_activation(conv_3)
            conv_3 = tf.nn.max_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("fc"):
            final_fc = self.blocks.fc(conv_3, 15360, 10)

        with tf.variable_scope("accuracy"):
            self.logits = final_fc
            label_count = 10
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.labels)
            self.classification_loss = tf.reduce_sum(classification_loss)

        # dropout_prob = 0.5
        # input_frequency_size = 40
        # input_time_size = 98
        # fingerprint_4d = tf.reshape(self.input,
        #                             [-1, input_time_size, input_frequency_size, 1])
        # with tf.device("/cpu:0"):
        #     tf.summary.image("mfcc", fingerprint_4d)
        # first_filter_width = 8
        # first_filter_height = 20
        # first_filter_count = 64
        # first_weights = tf.Variable(
        #     tf.truncated_normal(
        #         [first_filter_height, first_filter_width, 1, first_filter_count],
        #         stddev=0.01))
        # first_bias = tf.Variable(tf.zeros([first_filter_count]))
        # first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
        #                           'SAME') + first_bias
        # first_relu = tf.nn.relu(first_conv)
        #
        # first_dropout = tf.nn.dropout(first_relu, dropout_prob)
        #
        # max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        # second_filter_width = 4
        # second_filter_height = 10
        # second_filter_count = 64
        # second_weights = tf.Variable(
        #     tf.truncated_normal(
        #         [
        #             second_filter_height, second_filter_width, first_filter_count,
        #             second_filter_count
        #         ],
        #         stddev=0.01))
        # second_bias = tf.Variable(tf.zeros([second_filter_count]))
        # second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
        #                            'SAME') + second_bias
        # second_relu = tf.nn.relu(second_conv)
        #
        # second_dropout = tf.nn.dropout(second_relu, dropout_prob)
        #
        # second_conv_shape = second_dropout.get_shape()
        # second_conv_output_width = second_conv_shape[2]
        # second_conv_output_height = second_conv_shape[1]
        # second_conv_element_count = int(
        #     second_conv_output_width * second_conv_output_height *
        #     second_filter_count)
        # flattened_second_conv = tf.reshape(second_dropout,
        #                                    [-1, second_conv_element_count])
        #
        # final_fc_weights = tf.Variable(
        #     tf.truncated_normal(
        #         [second_conv_element_count, label_count], stddev=0.01))
        # final_fc_bias = tf.Variable(tf.zeros([label_count]))
        # final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

        with tf.variable_scope("accuracy"):
            self.logits = final_fc
            one_hot_labels = tf.one_hot(self.labels, label_count)
            prediction = tf.argmax(self.logits, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, prediction), dtype=tf.float32))
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=one_hot_labels)
            self.classification_loss = tf.reduce_mean(classification_loss)

        with tf.variable_scope("decay"):
            cost = []
            for weight in self.blocks.weights:
                cost.append(tf.nn.l2_loss(weight))
            decay = 0.0003 * tf.reduce_sum(cost)

        with tf.variable_scope("all_losses"):
            self.loss = self.classification_loss
            with tf.device("/cpu:0"):
                # tf.summary.scalar("decay", decay)
                tf.summary.scalar("classification_loss", self.classification_loss)
                tf.summary.scalar("total_loss", self.loss)

        with tf.variable_scope('classification_gradient'):
            boundaries = [15000, 18000]
            values = [0.001, 0.0001, 0.00001]

            with tf.device("/cpu:0"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
                tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
