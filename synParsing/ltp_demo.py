import os
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
print('\n'.join(sents))


# 分词
from pyltp import Segmentor
LTP_DATA_DIR = '/Users/cquan/Documents/model/ltp_data_v3.4.0'  # 分词模型路径，模型名称为`cws.model`
cws_model_path=os.path.join(LTP_DATA_DIR, 'cws.model')
segmentor = Segmentor()
segmentor.load(cws_model_path)
words=segmentor.segment('熊高雄你吃饭了吗')
print(type(words))
print('\t'.join(words))
segmentor.release()


# 词性标注
from pyltp import Postagger
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
postagger = Postagger()              # 初始化实例
postagger.load(pos_model_path)       # 加载模型
words = ['元芳', '你', '怎么', '看']   # 分词结果
postags = postagger.postag(words)    # 词性标注
print('\t'.join(postags))
postagger.release()  # 释放模型


# 命名实体识别
from pyltp import NamedEntityRecognizer
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
recognizer = NamedEntityRecognizer()   # 初始化实例
recognizer.load(ner_model_path)        # 加载模型
words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
netags = recognizer.recognize(words, postags)  # 命名实体识别
print('\t'.join(netags))
recognizer.release()  # 释放模型


# 依存句法分析
from pyltp import Parser
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
parser = Parser()                    # 初始化实例
parser.load(par_model_path)          # 加载模型
words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
arcs = parser.parse(words, postags)  # 句法分析
print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
parser.release()  # 释放模型


# 语义角色标注
from pyltp import SementicRoleLabeller
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
labeller = SementicRoleLabeller()    # 初始化实例
labeller.load(srl_model_path)        # 加载模型
words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
# arcs 使用依存句法分析的结果
roles = labeller.label(words, postags, arcs)  # 语义角色标注
# 打印结果
for role in roles:
    print(role.index, "".join(
        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
labeller.release()  # 释放模型


"""
可视化依存句法分析
"""
import jieba
sent = '2018年7月26日，华为创始人任正非向5G极化码（Polar码）之父埃尔达尔教授举行颁奖仪式，表彰其对于通信领域做出的贡献。'

jieba.add_word('Polar码')
jieba.add_word('5G极化码')
jieba.add_word('埃尔达尔')
jieba.add_word('之父')
words = list(jieba.cut(sent))
print(words)

postagger = Postagger()               # 初始化实例
postagger.load(pos_model_path)        # 加载模型
postags = postagger.postag(words)     # 词性标注
postagger.release()                   # 释放模型

parser = Parser()                     # 初始化实例
parser.load(par_model_path)           # 加载模型
arcs = parser.parse(words, postags)   # 依存句法分析
parser.release()                      # 释放模型
rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
relation = [arc.relation for arc in arcs]  # 提取依存关系
heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语

for i in range(len(words)):
    print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')


# from graphviz import Digraph
# g = Digraph('测试图片')
# g.node(name='Root')
# for word in words:
#     g.node(name=word)
#
# for i in range(len(words)):
#     if relation[i] not in ['HED']:
#         g.edge(words[i], heads[i], label=relation[i])
#     else:
#         if heads[i] == 'Root':
#             g.edge(words[i], 'Root', label=relation[i])
#         else:
#             g.edge(heads[i], 'Root', label=relation[i])

# g.view()


"""
利用networkx绘制句法分析结果
"""
import networkx as nx
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体

G = nx.Graph()  # 建立无向图G

# 添加节点
for word in words:
    G.add_node(word)

G.add_node('Root')

# 添加边
for i in range(len(words)):
    G.add_edge(words[i], heads[i])

source = '5G极化码'
target1 = '任正非'
distance1 = nx.shortest_path_length(G, source=source, target=target1)
print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target1, distance1))

target2 = '埃尔达尔'
distance2 = nx.shortest_path_length(G, source=source, target=target2)
print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target2, distance2))

nx.draw(G, with_labels=True)
plt.show()
# plt.savefig("undirected_graph.png")
