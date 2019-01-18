import tensorflow as tf

# Equal
v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
# v = tf.Variable(tf.constant(1.0, shape=[1], name='v'))
print(v)

# Right
with tf.variable_scope('foo', reuse=None):
    v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))
    print(v1)

# ValueError: Variable foo/v1 already exists, disallowed. Did you mean to set reuse=True ...
# with tf.variable_scope('foo', reuse=None):
#     v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))
#     print(v1)

# Right
with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))
    print(v1)

# ValueError: Variable fff/v3 does not exist, or was not created with tf.get_variable(). Did you mean...
# with tf.variable_scope('bar', reuse=True):
#     v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))

# Right
with tf.variable_scope('bar', reuse=None):
    v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))
    print(v1)

print('-----------')

with tf.variable_scope('a'):
    a = tf.Variable(initial_value=[1], name='a')
    print(a)

# with tf.name_scope('a'):
#     a = tf.Variable(initial_value=[1], name='a')
#     print(a)

with tf.variable_scope('a'):
    a = tf.Variable(initial_value=[1], name='a')
    print(a)
