import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载测试数据的读写工具包，加载测试手写数据，目录MNIST_data是用来存放下载网络上的训练和测试数据的
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Tensorflow 依赖于一个高效的 C++ 后端来进行计算。与后端的这个连接叫做 session。
# 一般而言，使用 TensorFlow 程序的流程是先创建一个图，然后在 session 中启动它。
# 为了在 Python 中进行高效的数值计算，我们通常会使用像 NumPy 一类的库，将一些诸如矩阵乘法的耗时操作在 Python 环境的外部来计算，
# 这些计算通常会通过其它语言并用更为高效的代码来实现。但遗憾的是，每一个操作切换回Python环境时仍需要不小的开销。
# TensorFlow 也是在 Python 外部完成其主要工作，但是进行了改进以避免这种开销。
# 其并没有采用在 Python 外部独立运行某个耗时操作的方式，而是先让我们描述一个交互操作图，然后完全将其运行在 Python 外部。
# 这与 Theano 或 Torch 的做法类似。
import tensorflow as tf
# 创建一个交互式的Session
sess = tf.InteractiveSession()

# 我们通过为输入图像和目标输出类别创建节点，来开始构建计算图
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# 为了创建这个模型，我们需要创建大量的权重和偏置项。
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是 ReLU 神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
# 为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。
def weight_variable(shape):
    """
    使用卷积神经网络会有很多权重和偏置需要创建,我们可以定义初始化函数便于重复使用
    这里我们给权重制造一些随机噪声避免完全对称,使用截断的正态分布噪声,标准差为0.1
    :param shape: 需要创建的权重Shape
    :return: 权重Tensor
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    偏置生成函数,因为激活函数使用的是ReLU,我们给偏置增加一些小的正值(0.1)避免死亡节点(dead neurons)
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# TensorFlow 在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用 vanilla 版本。
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
# 我们的池化用简单传统的2x2大小的模板做 max pooling。为了代码更简洁，我们把这部分抽象成一个函数。
def conv2d(x, w):
    """
    卷积层接下来要重复使用,tf.nn.conv2d是Tensorflow中的二维卷积函数,
    :param x: 输入[batch, in_height, in_width, in_channels]，第一个卷积层[batch数量，高度，宽度，通道数] = [-1, 28, 28, 1]
    :param w: 卷积核，[filter_height, filter_width, in_channels, out_channels],
              第一个卷积层[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] = [5, 5, 1, 32]
        strides:代表卷积模板移动的步长,都是1代表不遗漏的划过图片的每一个点.
                其为长度为4的一阶张量，并且要求strides[0]=strides[3]=1
                strides[1]，strides[2]决定卷积核在输入图像in_hight，in_width方向的滑动步长
        padding:代表边界处理方式, 目前有两种方式：
                一种是SAME，代表输入输出同尺寸，表示补齐操作后（在原始图像周围补充0）
                一种是VALID，补齐操作后，进行卷积过程中，原始图片中右边或者底部的像素数据可能出现丢弃的情况
    :return:
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    tf.nn.max_pool是TensorFLow中最大池化函数.我们使用2x2最大池化，因为希望整体上缩小图片尺寸,因而池化层的strides设为横竖两个方向为2步长
    :param x: 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
        ksize: 是一个长度不小于4的整型数组，每一位上的值对应于输入数据张量中每一维的窗口对应值
               池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 现在我们可以开始实现第一层了，它由一个卷积接一个 max pooling 完成
# 第一个卷积层  [5, 5, 1, 32]代表 卷积核尺寸为5x5,1个通道,32个不同卷积核
# 创建滤波器权值-->加偏置-->卷积-->池化
w_conv1 = weight_variable([5, 5, 1, 32])
# 而对于每一个输出通道都有一个对应的偏置量
b_conv1 = bias_variable([32])
# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
# 因为输入的时候x是一个[None,784]，有与reshape的输入项shape是[-1,28,28,1]，后续三个维度数据28,28,1相乘后得到784
# 所以，-1值在reshape函数中的特殊含义就可以映射程None。即输入图片的数量batch。
# 因为是灰度图所以这里的通道数为1，如果是 rgb 彩色图，则为3
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行 max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积池化层
# 卷积核依旧是5x5 通道为32，有64个不同的卷积核
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# h_pool2的大小为7x7x64 转为1-D 然后做FC层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层
# 用来减轻过拟合,通过一个placeholder传入keep_prob比率控制
# 在训练中,我们启用 dropout，随机丢弃一部分节点的数据来减轻过拟合
# 预测时则关闭 dropout，保留全部数据追求最佳性能
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax 层
# 将Dropout层的输出连接到一个Softmax层,得到最后的概率输出
W_fc2 = weight_variable([1024, 10])  # MNIST只有10种输出可能
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义损失函数,依旧使用交叉熵，同时定义优化器
# 我们用更加复杂的 ADAM 优化器来做梯度最速下降，learning rate = 1e-4
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 变量需要通过 seesion 初始化后，才能在session中使用。
sess.run(tf.global_variables_initializer())

# Tensorboard相关操作，用于记录训练过程，生成训练log，﻿供Tensorboard查看
# 然后可以通过执行tensorboard --logdir=/tmp/logs启动tensorboard服务，在本地浏览器访问http://127.0.0.1:6006就可以看了
# 添加一个Tensorboard里要观察的标量
tf.summary.scalar('cross_entropy', cross_entropy)
# 为了释放TensorBoard所使用的事件文件（events file），所有的即时数据（在这里只有一个）都要在图表构建阶段合并至一个操作（op）中
summary_op = tf.summary.merge_all()
# 实例化一个tf.train.SummaryWriter，用于写入包含了图表本身和即时数据具体值的事件文件
summary_writer = tf.summary.FileWriter('/tmp/logs', tf.get_default_graph())

# 然后开始训练模型，这里我们让模型循环训练1000次！
with tf.device('/gpu:0'):
    for i in range(2000):
        # 该循环的每个步骤中，我们都会随机抓取训练数据中的50个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行 train_step
        # 返回的batch其实是一个列表，元素0表示图像数据，元素1表示标签值
        batch = mnist.train.next_batch(50)

        # 每100次迭代输出一次日志
        # 这段if也可以放在train_step.run后面，放在这里是想测量从一开始的准确率，包括随机初始化后的第一次准确率
        if i % 10 == 0:
            # 在feed_dict中加入额外的参数keep_prob来控制 dropout 比例
            # 由于是测量训练时候的准确率，这里的dropout比例设为1，即关闭dropout，保留全部数据
            # 注意这里还是用同样的一批训练数据进行准确率的计算
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # 每次运行summary_op时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件读写器（writer）的add_summary()函数
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            summary_writer.add_summary(summary_str, i)

        # 此步主要是用来训练W和bias用的。收敛后，就等于W和bias都训练好了
        # 这里把dropout比率设为0.5，即dropout层丢失一半的神经元
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# 保存训练好的模型，格式为ckpt
# 由于TensorFlow 的版本一直在更新, 保存模型的方法也发生了改变
# 在python 环境,和在C++ 环境(移动端) 等不同的平台需要的模型文件也是不也一样的
# 模型有这几种格式： .meta .data .ckpt .pb
# .meta包含metagraph，.pb是C++环境要用的
saver = tf.train.Saver()  # defaults to saving all variables
saver.save(sess, r'./model/model.ckpt')  # The directory must be exist

# 将模型应用到测试集上，计算模型在测试数据集上面的准确率
# 在最终测试集上的准确率大概是99.2%
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
