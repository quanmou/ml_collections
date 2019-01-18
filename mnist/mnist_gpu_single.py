import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

x_image = tf.placeholder(tf.float32, shape=[None, 784])
y_class = tf.placeholder(tf.float32, shape=[None, 10])
x_ = tf.reshape(x_image, [-1, 28, 28, 1])

w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_, w_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, [1, 1, 1, 1], 'VALID') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

h_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])

w_fc1 = tf.Variable(tf.truncated_normal([5 * 5 * 16, 120], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
h_fc1 = tf.nn.relu(tf.matmul(h_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)

w_fc3 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_ = tf.matmul(h_fc2, w_fc3) + b_fc3

init_op = tf.global_variables_initializer()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_class, logits=y_))
train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)

predict = tf.equal(tf.argmax(y_class, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

saver = tf.train.Saver()

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('train_accuracy', accuracy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(r'./logs', tf.get_default_graph())

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(10000):
            batch = mnist.train.next_batch(128)
            train_dict = {x_image: batch[0], y_class: batch[1], keep_prob: 0.9}
            eval_dict = {x_image: batch[0], y_class: batch[1], keep_prob: 1.0}

            if step % 50 == 0:
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict=eval_dict)
                print("step: %s, loss: %s, acc: %s" % (step, loss, acc))
                saver.save(sess, r'./model/lenet.ckpt', global_step=step)

                summary_str = sess.run(summary_op, feed_dict=eval_dict)
                summary_writer.add_summary(summary_str, global_step=step)

            sess.run(train_op, feed_dict=train_dict)

    with tf.Session() as sess:
        ckpt_state = tf.train.get_checkpoint_state(r'./model')
        model_path = os.path.join(r'./model', os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess, model_path)
        print("test accuracy %g" % accuracy.eval(feed_dict={x_image: mnist.test.images, y_class: mnist.test.labels, keep_prob: 1.0}))
