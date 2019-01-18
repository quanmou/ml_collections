import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# init_op = tf.initialize_all_variables()  # Old function, deprecated
init_op = tf.initializers.global_variables()  # Returns an operation
# init_op = tf.global_variables_initializer()  # Optional function, the same as tf.initializers.global_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

with tf.variable_scope('scope1', reuse=tf.AUTO_REUSE) as scope1:
    x1 = tf.Variable(tf.ones([1]), name='x1')
    x2 = tf.Variable(tf.zeros([1]), name='x1')
    y1 = tf.get_variable('y1', initializer=1.0)
    y2 = tf.get_variable('y1', initializer=0.0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(x1.name, x1.eval())
        print(x2.name, x2.eval())
        print(y1.name, y1.eval())
        print(y2.name, y2.eval())
