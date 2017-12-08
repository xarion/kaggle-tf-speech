import tensorflow as tf

from dataset import Dataset
from model import Model
from submission_writer import SubmissionWriter

CHECKPOINT_STEP = 1000
CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_NAME = "model"
tf.logging.set_verbosity(tf.logging.INFO)

with tf.Session() as sess:
    data = Dataset(split="submission", batch_size=128)
    model = Model(data=data)

    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    if latest_checkpoint:
        tf.logging.info("loading from checkpoint file: " + latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
    else:
        tf.logging.error("checkpoint not found")

    submission_file = SubmissionWriter("submission", data.competition_labels)
    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            predictions, file_names, = sess.run([model.prediction, model.input_file_names])
            submission_file.add_records(predictions, file_names)

    except tf.errors.OutOfRangeError:
        tf.logging.info('Submission file created -- epoch limit reached')
    submission_file.close()
    coord.request_stop()
    coord.wait_for_stop()
    sess.close()
