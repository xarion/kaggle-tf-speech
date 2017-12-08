import tensorflow as tf

from blocks import Blocks


class Model:
    def __init__(self, data):
        self.input = data.inputs
        self.labels = data.labels
        self.input_file_names = data.file_names
        self.blocks = Blocks()

        self.input = tf.reshape(self.input, [-1, 98, 40, 1])

        with tf.variable_scope("conv_1"):
            conv_1 = self.blocks.conv2d(self.input, [8, 20, 1, 64])
            conv_1 = self.blocks.normalized_relu_activation(conv_1)
            conv_1_max_pool = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_1_avg_pool = tf.nn.avg_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_1 = tf.concat([conv_1_max_pool, conv_1_avg_pool], 3)

        with tf.variable_scope("conv_2"):
            conv_2 = self.blocks.conv2d(conv_1, [4, 10, 128, 128])
            conv_2 = self.blocks.normalized_relu_activation(conv_2)
            conv_2_max_pool = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_2_avg_pool = tf.nn.avg_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_2 = tf.concat([conv_2_max_pool, conv_2_avg_pool], 3)

        with tf.variable_scope("conv_3"):
            conv_3 = self.blocks.conv2d(conv_2, [2, 5, 256, 256])
            conv_3 = self.blocks.normalized_relu_activation(conv_3)
            conv_3_max_pool = tf.nn.max_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_3_avg_pool = tf.nn.avg_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_3_global_average = tf.reduce_mean(conv_3, axis=[1, 2])
            conv_3_global_variance = self.blocks.reduce_var(conv_3, axis=[1, 2])

            conv_3 = tf.concat([conv_3_max_pool, conv_3_avg_pool], 3)
            conv_3 = tf.reshape(conv_3, [-1, 15360 * 2])
            conv_3 = tf.concat([conv_3, conv_3_global_average, conv_3_global_variance], 1)

        with tf.variable_scope("fc"):
            final_fc = self.blocks.fc(conv_3, 15360 * 2 + 256 + 256, data.number_of_labels)

        with tf.variable_scope("accuracy"):
            self.logits = final_fc
            self.prediction = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.prediction), dtype=tf.float32))
            dense_labels = tf.one_hot(self.labels, 12, on_value=0.9, off_value=0.0091)
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=dense_labels)

            self.classification_loss = tf.reduce_mean(classification_loss)

        with tf.variable_scope("confusion_matrix"):
            self.confusion_matrix = tf.confusion_matrix(self.labels, self.prediction)

        with tf.variable_scope("decay"):
            cost = []
            for weight in self.blocks.weights:
                cost.append(tf.nn.l2_loss(weight))
            decay = 0.003 * tf.reduce_sum(cost)

        with tf.variable_scope("all_losses"):
            self.loss = self.classification_loss + decay
            tf.summary.scalar("decay", decay)
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope('classification_gradient'):
            boundaries = [7000, 10000]
            values = [0.001, 0.0001, 0.00001]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
