import tensorflow as tf

from data import Data

with tf.Session() as sess:
    data = Data()
    writer = tf.summary.FileWriter("summaries")

    merged = tf.summary.merge_all()

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)
    summaries, = sess.run([merged])

    writer.add_summary(summaries)
    writer.flush()
    writer.close()
    coord.request_stop()
    coord.wait_for_stop()
    sess.close()
