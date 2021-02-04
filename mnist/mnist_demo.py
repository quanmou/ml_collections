import cv2
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# path = "/Users/apple/Desktop"
path = os.path.dirname(os.path.abspath(__file__))

# 读取图片
img = cv2.imread(os.path.join(path, "images/1.png"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# 转换大小
img_shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

# 转灰度图
img_grey = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey, cmap='gray')
plt.show()

# 像素归一化
def minmaxNormalization(image):
    image = 255 - image
    Max = image.max()
    Min = image.min()
    return (image - Min) / (Max - Min)
img_norm = minmaxNormalization(img_grey)


# 模型架构
import tensorflow as tf

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

# 预测
y_hat = tf.nn.softmax(y_)

# 加载模型
saver = tf.train.Saver()
# 创建一个交互式的Session
sess = tf.InteractiveSession()
with tf.device('/cpu:0'):
    ckpt_state = tf.train.get_checkpoint_state(os.path.join(path, r'./model'))
    model_path = os.path.join(os.path.join(path, r'./model'), os.path.basename(ckpt_state.model_checkpoint_path))
    saver.restore(sess, model_path)
    prob = sess.run(y_hat, feed_dict={x_image: [img_norm.reshape(-1)], y_class: [[0]*10]})
    for i in range(10):
        print("%d: %.3f" % (i, round(prob[0][i], 3)))
