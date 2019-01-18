import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 128
seq_len = 28
input_size = 28
hidden_size = 64
output_size = 10

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_ = tf.reshape(x, [-1, seq_len, input_size])
x_ = tf.transpose(x_, [1, 0, 2])       # transpose to shape=(seq_len, batch_size, input_size)
x_ = tf.reshape(x_, [-1, input_size])  # reshape to shape=(seq_len * batch_size, input_size)
seqs = tf.split(x_, seq_len, 0)        # list (len=seq_len) of tensors, each tensor with shape=(batch_size, input_size)


def lstm_o(seq_list, batch_size, hidden_size, output_size):
    """
    Use official lstm function to generate output
    :param seq_list: A list of tensor with shape (batch_size, input_size)
    :param batch_size: Batch size
    :param hidden_size: Dimension of the hidden variable
    :param output_size: Dimension of the output classes
    :return: output
    """
    weight = tf.Variable(tf.truncated_normal([hidden_size, output_size]))  # output weight
    bias = tf.Variable(tf.zeros(shape=[output_size]))  # output biases
    outputs = list()

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    state = (tf.zeros([batch_size, hidden_size]),) * 2  # state = (c, h)

    with tf.variable_scope("myrnn2") as scope:
        for i, seq_input in enumerate(seq_list):
            if i > 0:
                scope.reuse_variables()   # I think this step is useless
            h, state = lstm_cell(seq_input, state)
            outputs.append(h)

    final = tf.matmul(outputs[-1], weight) + bias  # Only use the h from last sequence to compute y
    return final


def lstm_m(seq_list, batch_size, hidden_size, output_size):
    """
    Manually create a lstm network to generate output
    """
    input_size = seq_list[0].shape.dims[1].value
    # Parameters:
    # Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([input_size, hidden_size], -0.1, 0.1))
    iw = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, hidden_size]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([input_size, hidden_size], -0.1, 0.1))
    fw = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, hidden_size]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([input_size, hidden_size], -0.1, 0.1))
    cw = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, hidden_size]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([input_size, hidden_size], -0.1, 0.1))
    ow = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, hidden_size]))
    # Classifier weights and biases
    yw = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
    yb = tf.Variable(tf.zeros([output_size]))

    # Definition of the cell computation
    def lstm_cell(x, state):
        c, h = state
        input_gate = tf.sigmoid(tf.matmul(x, ix) + tf.matmul(h, iw) + ib)
        forget_gate = tf.sigmoid(tf.matmul(x, fx) + tf.matmul(h, fw) + fb)
        update = tf.tanh(tf.matmul(x, cx) + tf.matmul(h, cw) + cb)
        new_c = forget_gate * c + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(x, ox) + tf.matmul(h, ow) + ob)
        new_h = output_gate * tf.tanh(new_c)
        new_state = (new_c, new_h)
        return new_h, new_state

    # Unrolled LSTM loop
    outputs = list()
    state = (tf.zeros([batch_size, hidden_size]),) * 2  # state also can be tf.Variable

    for seq_input in seq_list:
        h, state = lstm_cell(seq_input, state)
        outputs.append(h)
    logits = tf.matmul(outputs[-1], yw) + yb
    return logits


y_ = lstm_o(seqs, batch_size, hidden_size, output_size)
# y_ = lstm_m(seqs, batch_size, hidden_size, output_size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

with tf.device('/gpu:0'):
    for step in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % 50 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Step: %s, Mini batch loss: %.6f, Train accuracy: %.5f" % (step, loss, acc))

# Calculate accuracy for 128 mnist test images
test_data = mnist.test.images[:batch_size]
test_label = mnist.test.labels[:batch_size]
print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
