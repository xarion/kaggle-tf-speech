import tensorflow as tf

from dataset import Dataset
from model import Model

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        data = Dataset(split="training", batch_size=128)
    with tf.device("/gpu:0"):
        model = Model(data=data)

    writer = tf.summary.FileWriter("summaries", sess.graph)

    merged = tf.summary.merge_all()

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    while step < 20000:
        m, _, loss, step, = sess.run([merged,
                                      model.training_step,
                                      model.loss,
                                      model.global_step])

        writer.add_summary(m, step)
    writer.flush()
    writer.close()
    coord.request_stop()
    coord.wait_for_stop()
    sess.close()
