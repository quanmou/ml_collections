import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets('./MNIST_data', one_hot=True)

# Training Parameters
learning_rate = 1e-3
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))  # shape=(2*128, 10)
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))  # shape=(10,)
}


def BiRNN(x, weights, biases):
    # Prepare data shape to match 'rnn' function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def stacked_bidirectional_rnn(cell, rnn_size, num_layers, inputs, seq_lengths, batch_size):
    """
    multi layer bidirectional rnn
    :param cell: RNN class, e.g. LSTMCell
    :param rnn_size: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int
    :return: the output of last layer bidirectional rnn with concatenating
    这里用到几个tf的特性
    1. tf.variable_scope(None, default_name="bidirectional-rnn")使用default_name
    的话,tf会自动处理命名冲突
    """
    # TODO: add time_major parameter, and using batch_size = tf.shape(inputs)[0], and more assert
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        # 恶心的很.如果不在这加的话,会报错的.
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = cell(rnn_size)
            rnn_cell_bw = cell(rnn_size)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs


lstm_cell = tf.contrib.rnn.BasicLSTMCell
outputs = stacked_bidirectional_rnn(lstm_cell, num_hidden, 2, X, [timesteps]*batch_size, batch_size)  # shape=(128, 28, 256)
logits_mul = tf.matmul(tf.reshape(outputs, [-1, outputs.shape[-1]]), weights['out'])      # shape=(128*28, 10)
logits_reshape = tf.reshape(logits_mul, [*outputs.shape[:-1], weights['out'].shape[-1]])  # shape=(128, 28, 10)
logits_reduce = tf.reduce_mean(logits_reshape, axis=1)    # shape=(128, 10)
logits = logits_reduce + biases['out']                    # shape=(128, 10)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    print("Optimization Finished")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
