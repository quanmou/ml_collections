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

class w2v_model:
    def __init__(self):
        model_file = '/datadrive/data/models/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    def vector_similarity(self, s1, s2):
        def sentence_vector(s):
            words = jieba.lcut(s)
            v = np.zeros(64)
            for word in words:
                v += self.model[word]
            v /= len(words)
            return v

        v1, v2 = sentence_vector(s1), sentence_vector(s2)
        return np.dot(v1, v2) / (norm(v1) * norm(v2))


if __name__ == "__main__":
    s1 = "你在干嘛呢"
    s2 = "你在干什么呢"
    print(jaccard_similarity(s1, s2))
    print(tf_similarity(s1, s2))
    print(tfidf_similarity(s1, s2))
    s1 = '你在干嘛'
    s2 = '你正做什么'

    s1 = """工作经验要求，2年以上
    1、全日制大学本科
    2、有银行或互联网金融开发测试经验，从事过互联网支付、互联网账户、票据等业务的测试工作
    3、2年以上测试实施经验，熟悉常用测试工具（Postman、Jmeter、LoadRunner等），熟悉测试管理工具（TFS、JIRA、禅道等）
    4、性格开朗、责任心强，善于沟通，有团队精神
    5、计算机专业优先、有自动化测试经验优先、银行相关项目经验优先
    6、年龄30以下"""

    s2 = """外场测试工程师
    1.工作职责
    外场测试工作人员主要开展相关电气分系统及产品测试工作，职责主要包括：
    1）开展测试状态准备和设备自检，并进行测试记录填写；
    2）按测试细则规定开展各种测试状态下的设备连接，进行测试操作，记录测试数据并进行判读；
    3）向设计人员及时反馈测试过程中的技术、质量问题，并在设计指导下现场处理简单技术问题。
    2.任职要求
    1）具有电气类工科专业（微波专业佳）大专及以上学历和相关工作经验。相关工作经历未满3年者，一般应具备大学本科及以上学历；
    2）熟悉航天质量管理体系，熟悉保密管理要求；
    3）熟悉航天电气系统及产品测试流程；
    4）工作责任心强，有较强的协调沟通能力，能适应频繁出差工作；
    5）身体健康，年龄不超过35周岁；
    6）具有微波产品测试工作经历者优先考虑。"""

    w2v = w2v_model()
    print(w2v.vector_similarity(s1, s2))
