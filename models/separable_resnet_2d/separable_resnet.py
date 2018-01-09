import tensorflow as tf

from blocks import Blocks


class Model:
    def __init__(self, data, parameters):
        """
        :param training: If this is true some update ops and input pipeline will be created.
        The update ops are not necessarily used because training is True.
        """
        self.batch_size = parameters['batch_size']
        self.training = parameters["training"]
        self.label_on_value = parameters["label_on_value"]
        self.label_off_value = parameters["label_off_value"]
        self.decay_rate = parameters["decay_rate"]
        self.filter_size = parameters["filter_size"]
        self.stride_length = parameters["stride_length"]
        self.bigram_model = parameters["bigram_model"]
        self.num_bigrams = parameters["num_bigrams"]
        self.max_model_width = parameters["max_model_width"]
        self.global_avg_pooling = parameters["global_avg_pooling"]
        self.use_adam = parameters["use_adam"]
        self.sigmoid_unknown = parameters["sigmoid_unknown"]
        self.accuracy_regulated_decay = parameters["accuracy_regulated_decay"]
        self.loss_regulated_decay = parameters["loss_regulated_decay"]
        self.blocks = Blocks(self.training, parameters)
        self.model_width = parameters["model_width"]  # 32
        self.model_setup = map(lambda x: int(x.strip()), parameters["model_setup"].split(','))  # [0, 4, 4]
        self.input = data.inputs
        self.labels = data.labels
        self.input_file_names = data.file_names

        self.learning_rate = None
        self.logits = self.inference(self.input)

        self.top_1_accuracy = None
        self.loss = None
        self.train_step = None
        self.global_step = None
        self.one_hot_truth = None
        self.optimizer = None
        self.evaluation()
        self.optimize()
        # dummy variable that is temporarily ignored

    def inference(self, preprocessed_input):

        channels = self.model_width

        with tf.variable_scope("conv_1_1"):
            conv_1_1 = self.blocks.conv2d_with_filter(preprocessed_input,
                                                      filter_size=[20, 8],
                                                      input_channels=1,
                                                      output_channels=channels,
                                                      strides=1)
        residual_layer = conv_1_1
        input_channels = channels

        for residual_block_set in range(0, len(self.model_setup)):
            output_channels = input_channels * 2
            if self.max_model_width < output_channels:
                output_channels = self.max_model_width

            if self.model_setup[residual_block_set] == 0:
                residual_layer = tf.nn.max_pool(residual_layer, ksize=[1, self.filter_size, 1, 1],
                                                strides=[1, self.stride_length, 1, 1],
                                                padding="SAME")

            for residual_block in range(0, self.model_setup[residual_block_set]):
                with tf.variable_scope("conv_%d_%d" % (residual_block_set + 2, residual_block + 1)):
                    residual_layer = self.blocks.residual_separable(residual_layer,
                                                                    input_channels=input_channels,
                                                                    output_channels=output_channels,
                                                                    strides=self.stride_length if residual_block == 0 else 1,
                                                                    activate_before_residual=residual_block == 0)
                    input_channels = output_channels

        if self.bigram_model:
            with tf.variable_scope("bigram"):
                output_channels = self.num_bigrams
                residual_layer = self.blocks.conv2d(residual_layer,
                                                    filter_size=2,
                                                    input_channels=input_channels,
                                                    output_channels=output_channels,
                                                    strides=1)
                input_channels = output_channels

        if self.global_avg_pooling:
            with tf.variable_scope("global_pooling"):
                # global average pooling
                residual_layer = tf.reduce_max(residual_layer, [1, 2])
                residual_layer = self.blocks.batch_normalization(residual_layer)
                residual_layer = self.blocks.relu(residual_layer)
                # residual_layer = self.blocks.biased_fc(residual_layer,
                #                                                input_channels=input_channels,
                #                                                output_channels=input_channels)
                # residual_layer = self.blocks.batch_normalization(residual_layer)
                # residual_layer = self.blocks.relu(residual_layer)
        else:
            with tf.variable_scope("flatten"):
                shape = residual_layer.shape
                dims = shape[1].value * shape[3].value
                residual_layer = tf.reshape(residual_layer, shape=[-1, dims])
                output_channels = dims
                input_channels = output_channels
                residual_layer = self.blocks.batch_normalization(residual_layer)
                residual_layer = self.blocks.relu(residual_layer)

        with tf.variable_scope("output"):
            logits = self.blocks.biased_fc(residual_layer,
                                           input_channels=input_channels,
                                           output_channels=12)
            self.freeze_layer = logits
        return logits

    def optimize(self):
        with tf.variable_scope('loss'):
            if not self.sigmoid_unknown:
                self.one_hot_truth = tf.squeeze(tf.one_hot(self.labels, 12,
                                                           on_value=self.label_on_value,
                                                           off_value=self.label_off_value))
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_truth)
                self.loss = tf.reduce_mean(cross_entropy)
            else:
                self.one_hot_truth = tf.squeeze(tf.one_hot(self.labels, 12,
                                                           on_value=self.label_on_value,
                                                           off_value=self.label_off_value))
                class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :11],
                                                                              labels=self.one_hot_truth[:, :11])
                class_results = tf.reduce_max(self.one_hot_truth[:, :11], axis=1)
                sigmoid_labels = (self.one_hot_truth[:, 11:]) * 0.97
                silent_unknown_cross_entropy = \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits[:, 11:],
                                                            labels=sigmoid_labels)

                self.loss = tf.reduce_mean(class_cross_entropy * class_results)
                self.loss += tf.reduce_mean(silent_unknown_cross_entropy)
                self.loss += (1. - self.accuracy)

            if self.accuracy_regulated_decay:
                self.loss = self.loss + (1 - self.accuracy) * self.decay()
            elif self.loss_regulated_decay:
                self.loss = self.loss + self.loss * self.decay()
            else:
                self.loss = self.loss + self.decay()
            tf.add_to_collection('losses', self.loss)
            tf.summary.scalar('loss_total', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.probabilities = tf.nn.softmax(self.logits)

            if self.use_adam:
                self.optimizer = tf.train.AdamOptimizer()
            else:
                boundaries = [25000, 35000]
                values = [0.1, 0.01, 0.001]

                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
                tf.summary.scalar('learning_rate', self.learning_rate)

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def evaluation(self):
        with tf.variable_scope('accuracy'):
            if not self.sigmoid_unknown:
                self.prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)
            else:
                class_predictions = tf.cast(tf.argmax(self.logits[:, :11], 1), tf.int32)
                silent_unknown_sigmoids = tf.nn.sigmoid(self.logits[:, 11:])
                silent_unknown_labels = tf.cast(tf.argmax(silent_unknown_sigmoids, 1), tf.int32) + 11
                accepted_silent_unknown_labels = tf.cast(
                    tf.greater_equal(tf.reduce_max(silent_unknown_sigmoids, axis=1), 0.2), tf.int32)
                self.prediction = \
                    class_predictions - (class_predictions * accepted_silent_unknown_labels) \
                    + silent_unknown_labels * accepted_silent_unknown_labels

            correct_prediction = tf.equal(self.prediction, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar('accuracy_top1', self.accuracy)
        with tf.variable_scope("confusion_matrix"):
            self.confusion_matrix = tf.confusion_matrix(self.labels, self.prediction)

    def decay(self):
        """L2 weight decay loss."""
        costs = list()
        for var in self.blocks.get_decayed_variables():
            costs.append(tf.nn.l2_loss(var))

        decay = tf.reduce_sum(costs)
        tf.summary.scalar('decay', decay)

        return tf.multiply(self.decay_rate, decay)
