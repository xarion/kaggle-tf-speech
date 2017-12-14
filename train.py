import tensorflow as tf

from data.splitting_dataset import SplittingDataset
from model import Model

tf.logging.set_verbosity(tf.logging.INFO)


class Train:
    def __init__(self, parameters=None):
        self.parameters = parameters
        self.run_checks()
        self.session = tf.Session()
        with self.session.as_default():
            self.data = SplittingDataset(self.parameters)

            self.experiment_name = self.parameters['experiment_name']
            self.model = Model(data=self.data, parameters=self.parameters)
            self.summary_dir = self.parameters['master_folder'] + '/summaries/' + self.parameters['experiment_name'] + '/'
            self.checkpoint_dir = self.parameters['master_folder'] + '/checkpoints/' + self.parameters['experiment_name'] + '/'

            self.train_writer = tf.summary.FileWriter(self.summary_dir + "train", self.session.graph)
            self.validation_writer = tf.summary.FileWriter(self.summary_dir + "validation", self.session.graph,
                                                           flush_secs=20)
            self.merged_summaries = tf.summary.merge_all()

            self.saver = tf.train.Saver(max_to_keep=6, keep_checkpoint_every_n_hours=3)
            self.coord = tf.train.Coordinator()

    def initialize(self):
        self.session.run(tf.variables_initializer(tf.global_variables()))
        self.session.run(tf.variables_initializer(tf.local_variables()))

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            tf.logging.info("loading from checkpoint file: " + latest_checkpoint)
            self.saver.restore(self.session, latest_checkpoint)
        else:
            tf.logging.info("checkpoint not found")

        tf.train.start_queue_runners(sess=self.session, coord=self.coord)

    def run(self):
        step = 0
        last_step = 0
        try:
            while not self.coord.should_stop() and not (step >= 25000):
                # Train the model
                m, _, loss, step, labels, accuracy, = self.session.run([self.merged_summaries,
                                                                        self.model.training_step,
                                                                        self.model.loss,
                                                                        self.model.global_step,
                                                                        self.model.labels,
                                                                        self.model.accuracy])
                last_step = step
                tf.logging.info("step %d, loss %.6f, accuracy %.2f" % (step, loss, accuracy))

                self.train_writer.add_summary(m, step)

                if step % self.parameters['checkpoint_step'] == 0:
                    self.save_checkpoint(step)

                last_step = step

                # Do Validation sometimes
                if last_step % self.parameters['validation_step'] == 0:
                    m, accuracy, confusion_matrix = self.session.run([self.merged_summaries,
                                                                      self.model.accuracy,
                                                                      self.model.confusion_matrix],
                                                                     feed_dict={self.data.do_validate: True})
                    self.validation_writer.add_summary(m, global_step=step)
                    tf.logging.info("===== validation accuracy accuracy %.2f =====" % (accuracy))
                    tf.logging.info("\n" + str(confusion_matrix))

        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            if last_step:
                self.save_checkpoint(last_step)

    def save_checkpoint(self, step):
        self.saver.save(self.session, self.checkpoint_dir + "model", global_step=step)

    def finalize(self):
        self.validation_writer.flush()
        self.validation_writer.close()
        self.train_writer.flush()
        self.train_writer.close()
        self.coord.request_stop()
        self.coord.wait_for_stop()
        self.session.close()

    def train(self):
        self.initialize()
        self.run()
        self.finalize()

    def run_checks(self):
        assert self.parameters['experiment_name'] is not '' and self.parameters['experiment_name'] is not None, "Experiment name can not be empty"



