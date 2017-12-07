import tensorflow as tf


l = range(0, 4)
s = ["zero", "one", "two", "three"]
ss = tf.constant(["zero", "one", "two", "three"])
ll = tf.constant(l)
# lll = tf.expand_dims(ll, 0)
lq_1 = tf.train.input_producer(ll, shuffle=False)
lq_2 = tf.train.string_input_producer(ss, shuffle=False)

l_1, l_2 = tf.train.batch([lq_1.dequeue(), lq_2.dequeue()], batch_size=1024, num_threads=12)

# cd = tf.assert_equal(l_1, l_2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    queueRunners = tf.train.start_queue_runners(sess=sess, coord=coord)
    o_1, o_2, = sess.run([l_1, l_2])
    for i, st in zip(o_1, o_2):
        assert st == s[i], "dequeued elements are not equal. i = %d, s = %s" % (i, st)