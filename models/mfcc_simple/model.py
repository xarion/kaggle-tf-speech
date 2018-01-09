import tensorflow as tf

from blocks import Blocks


class Model:
    def __init__(self, data, parameters):
        self.parameters = parameters
        self.input = data.inputs
        self.labels = data.labels
        self.input_file_names = data.file_names
        self.blocks = Blocks()
        self.data = data

        self.models = {"default_mfcc_model": self.default_mfcc_model,
                       "unfinished_model": self.unfinished_model}
        self.models[self.parameters['model']]()

    def default_mfcc_model(self):

        assert self.parameters['mfcc_inputs']

        with tf.variable_scope("conv_1"):
            conv_1 = self.blocks.conv2d(self.input, [20, 8, 1, 64])
            conv_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_1 = self.blocks.normalized_relu_activation(conv_1)

        # with tf.variable_scope("conv_1_2"):
        #     conv_1 = self.blocks.conv2d(conv_1, [8, 20, 64, 64])
        #     conv_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        #     conv_1 = self.blocks.normalized_relu_activation(conv_1)

        with tf.variable_scope("conv_2"):
            conv_2 = self.blocks.conv2d(conv_1, [10, 4, 64, 128])
            conv_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
            conv_2 = self.blocks.normalized_relu_activation(conv_2)
        # with tf.variable_scope("conv_2_2"):
        #     conv_2 = self.blocks.conv2d(conv_2, [4, 10, 128, 256])
        #     conv_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        #     conv_2 = self.blocks.normalized_relu_activation(conv_2)

        with tf.variable_scope("fc"):
            final_fc = self.blocks.fc(conv_2, 30720, self.data.number_of_labels)

        with tf.variable_scope("accuracy"):
            self.logits = final_fc
            self.prediction = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.prediction), dtype=tf.float32))
            dense_labels = tf.one_hot(self.labels, 12)
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=dense_labels)

            self.classification_loss = tf.reduce_mean(classification_loss)

        with tf.variable_scope("confusion_matrix"):
            self.confusion_matrix = tf.confusion_matrix(self.labels, self.prediction)

        with tf.variable_scope("decay"):
            cost = []
            for weight in self.blocks.weights:
                cost.append(tf.nn.l2_loss(weight))
            decay = 0.0001 * tf.reduce_sum(cost)

        with tf.variable_scope("all_losses"):
            self.loss = self.classification_loss + decay
            tf.summary.scalar("decay", decay)
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope('classification_gradient'):
            boundaries = [20000]
            values = [0.001, 0.0001]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def unfinished_model(self):

        assert not self.parameters['mfcc_inputs']

        with tf.variable_scope("conv_1"):
            conv_1 = self.blocks.conv2d(self.input, [9, 1, 1, 32])
            conv_1 = self.blocks.normalized_relu_activation(conv_1)
            conv_1 = tf.nn.max_pool(conv_1, [1, 2, 1, 1], [1, 2, 1, 1], padding="VALID")

        with tf.variable_scope("conv_1_2"):
            conv_1 = self.blocks.conv2d(conv_1, [9, 1, 32, 64])
            conv_1 = self.blocks.normalized_relu_activation(conv_1)
            conv_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("conv_2"):
            conv_2 = self.blocks.conv2d(conv_1, [9, 1, 64, 128])
            conv_2 = self.blocks.normalized_relu_activation(conv_2)
            conv_2 = tf.nn.max_pool(conv_2, [1, 2, 1, 1], [1, 2, 1, 1], padding="VALID")

        with tf.variable_scope("conv_2_2"):
            conv_2 = self.blocks.conv2d(conv_2, [4, 10, 128, 256])
            conv_2 = self.blocks.normalized_relu_activation(conv_2)
            conv_2 = tf.nn.max_pool(conv_2, [1, 2, 1, 1], [1, 2, 1, 1], padding="VALID")

        with tf.variable_scope("fc"):
            final_fc = self.blocks.fc(conv_2, 61440, self.data.number_of_labels)

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
            decay = 0.01 * tf.reduce_sum(cost)

        with tf.variable_scope("all_losses"):
            self.loss = self.classification_loss + decay
            tf.summary.scalar("decay", decay)
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope('classification_gradient'):
            boundaries = [20000]
            values = [0.1, 0.01]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
