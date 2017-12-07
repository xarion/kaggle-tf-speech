import tensorflow as tf
import numpy as np

from dataset import Dataset
from model import Model

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        data = Dataset(split="training", batch_size=3)
    with tf.device("/gpu:0"):
        model = Model(data=data)

    writer = tf.summary.FileWriter("summaries", sess.graph)

    merged = tf.summary.merge_all()

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    # while step < 20000:
    m, _, loss, step, labels, accuracy, = sess.run([merged,
                                                    model.training_step,
                                                    model.loss,
                                                    model.global_step,
                                                    model.labels, model.accuracy])
    tf.logging.info("step %d, loss %.6f, accuracy %.2f" % (step, loss, accuracy))
    tf.logging.info(labels)
    writer.add_summary(m, step)
    writer.flush()
    writer.close()
    coord.request_stop()
    coord.wait_for_stop()
    sess.close()
