import tensorflow as tf

# 加载测试数据的读写工具包，加载测试手写数据，目录MNIST_data是用来存放下载网络上的训练和测试数据的
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 创建一个占位符，数据类型是float。x占位符的形状是[None，784]，即用来存放图像数据的变量，图像有多少张
# 是不关注的。但是图像的数据维度有784围。怎么来的，因为MNIST处理的图片都是28*28的大小，将一个二维图像
# 展平后，放入一个长度为784的数组中。 None表示此张量的第一个维度可以是任意长度的。
x = tf.placeholder(tf.float32, shape=[None, 784])  # x为特征

# 我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但 TensorFlow 有一个更好的方法来表示它们：Variable 。
# 一个 Variable 代表一个可修改的张量，存在在 TensorFlow 的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用 Variable 表示。
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 现在，我们可以实现我们的模型啦。只需要一行代码！
# 首先，我们用tf.matmul(X，W)表示x乘以W，对应之前等式里面的，这里x是一个2维张量拥有多个输入。然后再加上b，把和输入到tf.nn.softmax函数里面。
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 为了计算交叉熵，我们首先需要添加一个占位符用于输入正确值
# y_占位符的形状类似x，只是维度只有10，因为输出结果是0-9的数字，所以只有10种结构。
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # y_为label

# 计算交叉熵， y_是label真实值，y是预测值
# 用 tf.reduce_mean 计算张量的所有元素平均值。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的平均值。）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 现在我们知道我们需要我们的模型做什么啦，用TensorFlow来训练它是非常容易的。
# 在这里，我们要求 TensorFlow 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个交互式的Session
# 现在我们可以在一个 Session 里面启动我们的模型，并且初始化变量
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 然后开始训练模型，这里我们让模型循环训练1000次！
for _ in range(1000):
    # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行 train_step
    batch_x, batch_y = mnist.train.next_batch(100)
    loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y})
    if _ % 10 == 0:
        print('Step {0}, loss {1}'.format(_, loss))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# 评估我们的模型性能
# tf.argmax 是一个非常有用的函数，它能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签。比如 tf.argmax(y,1) 返回的是模型对于任一输入x预测到的标签值
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 最后，我们计算所学习到的模型在测试数据集上面的正确率
# 这个最终结果值应该大约是91%。事实上，这个结果很差
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
