import tensorflow as tf

a = "1/2/3/4/5/6/7/8/9"
tensor = tf.convert_to_tensor([a])
split = tf.string_split(tensor, '/')

checks = [tf.equal(split.values[i], tf.convert_to_tensor(a.split('/')[i])) for i in range(-8, 8)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(checks))
