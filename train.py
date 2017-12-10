import tensorflow as tf

from flags import FLAGS

from data.splitting_dataset import SplittingDataset
from model import Model

CHECKPOINT_STEP = 1000
CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_NAME = "model"
VALIDATION_STEP = 100
tf.logging.set_verbosity(tf.logging.INFO)


class Train:
    def __init__(self):
        self.session = tf.Session()

        self.data = SplittingDataset(training_batch_size=FLAGS.batch_size,
                                     validation_batch_size=FLAGS.validation_batch_size)

        model_config = dict()
        self.model = Model(data=self.data, model_config=model_config)
        self.train_writer = tf.summary.FileWriter("summaries/train", self.session.graph)
        self.validation_writer = tf.summary.FileWriter("summaries/validation", self.session.graph, flush_secs=20)
        self.merged_summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=6, keep_checkpoint_every_n_hours=3)
        self.coord = tf.train.Coordinator()

    def initialize(self):
        self.session.run(tf.variables_initializer(tf.global_variables()))
        self.session.run(tf.variables_initializer(tf.local_variables()))

        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
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
                if step % CHECKPOINT_STEP == 0:
                    self.saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
                last_step = step

                # Do Validation sometimes
                if last_step % VALIDATION_STEP == 0:
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
                self.saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=last_step)

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


def main(_):
    t = Train()
    t.train()


if __name__ == '__main__':
    tf.app.run()
