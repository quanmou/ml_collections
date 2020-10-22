# 文本相似性的几种计算方式


# 编辑距离
def edit_distance(s1, s2):
    l1, l2 = len(s1) + 1, len(s2) + 1
    matrix = [[0] * l2 for i in range(l1)]
    for i in range(1, l1):
        matrix[i][0] = i
    for i in range(1, l2):
        matrix[0][i] = i
    for i in range(1, l1):
        for j in range(1, l2):
            if s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1], matrix[i][j-1], matrix[i-1][j]) + 1
    return matrix[-1][-1]


strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]
target = '你在干啥'
results = list(filter(lambda x: edit_distance(x, target) <= 2, strings))
print(results)


# 杰拉德系数
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def add_space(s):
    return ' '.join(list(s))

def jaccard_similarity(s1, s2):
    # 在字符之间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


# TF系数
from scipy.linalg import norm
def tf_similarity(s1, s2):
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_similarity(s1, s2):
    s1, s2 = add_space(s1), add_space(s2)
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# word2vec
import gensim
import jieba
model_file = '/datadrive/data/models/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


s1 = "你在干嘛呢"
s2 = "你在干什么呢"
print(jaccard_similarity(s1, s2))
print(tf_similarity(s1, s2))
print(tfidf_similarity(s1, s2))
s1 = '你在干嘛'
s2 = '你正做什么'
print(vector_similarity(s1, s2))
