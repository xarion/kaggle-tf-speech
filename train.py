import tensorflow as tf

from data.splitting_dataset import SplittingDataset
from model import Model

CHECKPOINT_STEP = 1000
CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_NAME = "model"
tf.logging.set_verbosity(tf.logging.INFO)

with tf.Session() as sess:
    data = Dataset(split="training", batch_size=128)
    model = Model(data=data)

    train_writer = tf.summary.FileWriter("summaries/train", sess.graph)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=6, keep_checkpoint_every_n_hours=3)

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    if latest_checkpoint:
        tf.logging.info("loading from checkpoint file: " + latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
    else:
        tf.logging.info("checkpoint not found")

    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    last_step = 0
    try:
        while not coord.should_stop() and not (step >= 11000):
            m, _, loss, step, labels, accuracy, = sess.run([merged,
                                                            model.training_step,
                                                            model.loss,
                                                            model.global_step,
                                                            model.labels, model.accuracy])
            last_step = step
            tf.logging.info("step %d, loss %.6f, accuracy %.2f" % (step, loss, accuracy))
            train_writer.add_summary(m, step)
            if step % CHECKPOINT_STEP == 0:
                saver.save(sess, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
            last_step = step
    except tf.errors.OutOfRangeError:
        tf.logging.info('Done training -- epoch limit reached')
    finally:
        if last_step:
            saver.save(sess, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=last_step)

    train_writer.flush()
    train_writer.close()
    coord.request_stop()
    coord.wait_for_stop()
    sess.close()
