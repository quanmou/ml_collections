import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
tf.flags.DEFINE_string('gpu_list', '6,7', 'gpu list')
FLAGS = tf.flags.FLAGS
gpu_ids = FLAGS.gpu_list.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

x_image = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
w_fc1 = tf.Variable(tf.truncated_normal([400, 84], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[84]))
w_fc2 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))


def lenet(image, keep_pro=1):
    x_input = tf.reshape(image, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_input, w_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, [1, 1, 1, 1], 'VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    h_flat = tf.reshape(h_pool2, [-1, 400])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, w_fc1) + b_fc1)
    h_dropout = tf.nn.dropout(h_fc1, keep_pro)
    y_output = tf.matmul(h_dropout, w_fc2) + b_fc2
    return y_output


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


image_splits = tf.split(x_image, len(gpu_ids))
label_splits = tf.split(y_label, len(gpu_ids))
optimizer = tf.train.GradientDescentOptimizer(0.1)

y_list = []
loss_list = []
grad_list = []
for i, gpu_id in enumerate(gpu_ids):
    with tf.device('/gpu:%s' % gpu_id):
        image_split = image_splits[i]
        label_split = label_splits[i]
        y_ = lenet(image_split, keep_prob)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_split, logits=y_))
        gradient = optimizer.compute_gradients(loss)

        y_list.append(y_)
        loss_list.append(loss)
        grad_list.append(gradient)

mean_loss = tf.reduce_mean(tf.stack(loss_list), 0)
mean_grad = average_gradients(grad_list)
train_op = optimizer.apply_gradients(mean_grad)

predict = tf.equal(tf.argmax(y_label, 1), tf.argmax(tf.concat(y_list, 0), 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

tf.summary.scalar('cross_entropy', mean_loss)
tf.summary.scalar('train_accuracy', accuracy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(r'./logs', tf.get_default_graph())

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init_op)
    for step in range(1000):
        batch = mnist.train.next_batch(100)
        train_dict = {x_image: batch[0], y_label: batch[1], keep_prob: 0.9}
        eval_dict = {x_image: batch[0], y_label: batch[1], keep_prob: 1.0}

        if step % 10 == 0:
            loss_, acc_ = sess.run([mean_loss, accuracy], feed_dict=eval_dict)
            print("step: %s, loss: %s, acc: %s" % (step, loss_, acc_))
            saver.save(sess, r'./model/lenet.ckpt', global_step=step)
            summary_str = sess.run(summary_op, feed_dict=eval_dict)
            summary_writer.add_summary(summary_str, global_step=step)

        sess.run(train_op, feed_dict=train_dict)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    ckpt_state = tf.train.get_checkpoint_state(r'./model')
    model_path = os.path.join(r'./model', os.path.basename(ckpt_state.model_checkpoint_path))
    saver.restore(sess, model_path)
    print("Test accuracy: %s" % accuracy.eval({x_image: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))
