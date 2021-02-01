from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 训练集
train_images = mnist.train.images
train_labels = mnist.train.labels

# 验证集
validation_images = mnist.validation.images
validation_labels = mnist.validation.labels

# 测试集
test_images = mnist.test.images
test_labels = mnist.test.labels

# 打印训练集、验证集、测试集的数据条数
print(mnist.train.num_examples)         # 55000
print(mnist.validation.num_examples)    # 5000
print(mnist.test.num_examples)          # 10000

# 读取一条训练数据并打印
im = mnist.train.images[0]
print(im.shape)
print(im)


# 将训练数据reshape，并画出来
from matplotlib import pyplot as plt
im = im.reshape(28, 28)
print(im.shape)
plt.imshow(im, cmap='gray')
plt.show()

# 粗略的画出来
for r in im:
    print([int(c != 0) for c in r])

# 打印标签
print(mnist.train.labels[0])

# 多打印几个7的写法
indexes = [i for i, label in enumerate(train_labels) if label[7] == 1]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(25):
    img = train_images[indexes[i]].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 打印其他的数字看看
fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(25):
    img = mnist.train.images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 打印更多的数字
fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    img = mnist.train.images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 看看数据集中各个数字的数据量
X, Y = [], []
for i in range(10):
    x = i
    y = sum([1 for label in train_labels if label[i] == 1])
    X.append(x)
    Y.append(y)
    plt.text(x, y, '%s' % y, ha='center', va='bottom')

plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
plt.xticks(X)
plt.show()
