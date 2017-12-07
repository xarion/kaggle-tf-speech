import tensorflow as tf

from blocks import Blocks


class Model:
    def __init__(self, data):
        self.input = data.inputs
        self.labels = data.labels
        self.blocks = Blocks()

        self.input = tf.reshape(self.input, [-1, 98, 40, 1])

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
            final_fc = self.blocks.fc(conv_3, 15360, data.number_of_labels)

        with tf.variable_scope("accuracy"):
            self.logits = final_fc
            prediction = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, prediction), dtype=tf.float32))
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.labels)
            self.classification_loss = tf.reduce_mean(classification_loss)

        with tf.variable_scope("decay"):
            cost = []
            for weight in self.blocks.weights:
                cost.append(tf.nn.l2_loss(weight))
            decay = 0.0003 * tf.reduce_sum(cost)

        with tf.variable_scope("all_losses"):
            self.loss = self.classification_loss + decay
            tf.summary.scalar("decay", decay)
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope('classification_gradient'):
            boundaries = [15000, 18000]
            values = [0.001, 0.0001, 0.00001]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
