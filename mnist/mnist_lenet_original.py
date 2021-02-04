import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定义图像输入尺寸
x_image = tf.placeholder(tf.float32, shape=[None, 784])
y_class = tf.placeholder(tf.float32, shape=[None, 10])
x_ = tf.reshape(x_image, [-1, 28, 28, 1])

# 第一层：卷积层
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
conv1 = tf.nn.conv2d(x_, w_conv1, [1, 1, 1, 1], 'SAME') + b_conv1
h_conv1 = tf.nn.sigmoid(conv1)

# 第二层：池化层
h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

# 第三层：卷积层
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
conv2 = tf.nn.conv2d(h_pool1, w_conv2, [1, 1, 1, 1], 'VALID') + b_conv2
h_conv2 = tf.nn.relu(conv2)

# 第四层：池化层
h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

# 第五层：全连接层
h_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])  # 拉平
w_fc1 = tf.Variable(tf.truncated_normal([5 * 5 * 16, 120], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
h_fc1 = tf.nn.relu(tf.matmul(h_flat, w_fc1) + b_fc1)

# 第六层：全连接层
w_fc2 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

# 第七层：输出层
w_fc3 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_ = tf.matmul(h_fc2, w_fc3) + b_fc3

# 预测，计算损失
y_hat = tf.nn.softmax(y_)
cross_entropy_loss = -tf.reduce_sum(y_class * tf.log(y_hat))  # 计算交叉熵损失

# 定义优化器
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_class, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义模型保存器
saver = tf.train.Saver()
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('train_accuracy', accuracy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(r'./logs', tf.get_default_graph())

# 配置session运行参数
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 训练模型
with tf.device('/gpu:0'):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            batch = mnist.train.next_batch(128)
            train_dict = {x_image: batch[0], y_class: batch[1]}
            eval_dict = {x_image: batch[0], y_class: batch[1]}

            if step % 50 == 0:
                loss, acc = sess.run([cross_entropy_loss, accuracy], feed_dict=eval_dict)
                print("step: %s, loss: %s, acc: %s" % (step, loss, acc))
                saver.save(sess, r'./model/lenet.ckpt', global_step=step)

                summary_str = sess.run(summary_op, feed_dict=eval_dict)
                summary_writer.add_summary(summary_str, global_step=step)

            sess.run(train_op, feed_dict=train_dict)

# 测试模型
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        ckpt_state = tf.train.get_checkpoint_state(r'./model')
        model_path = os.path.join(r'./model', os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess, model_path)
        print("test accuracy %g" % accuracy.eval(feed_dict={x_image: mnist.test.images, y_class: mnist.test.labels}))
