import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# tf.set_random_seed(1)
batch_size = 128
n_input = 28
n_step = 28
n_hidden_unit = 128
n_class = 10

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_class])

weights = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_unit])),
    'out': tf.Variable(tf.random_normal([n_hidden_unit, n_class]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class]))
}


def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x_in = tf.matmul(x, weights['in'] + biases['in'])
    x_in = tf.reshape(x_in, [-1, n_step, n_hidden_unit])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unit, forget_bias=1.0,  state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    output, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    outputs = tf.unstack(tf.transpose(output, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


predict = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
train_op = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        for step in range(10000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([batch_size, n_step, n_input])
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
            if step % 50 == 0:
                loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
                print('step: %s, loss: %s, train accuracy: %s' % (step, loss_, acc_))
