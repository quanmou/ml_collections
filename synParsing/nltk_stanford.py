from nltk.tokenize import StanfordSegmenter
# from nltk.tokenize import StanfordTokenizer

segmenter = StanfordSegmenter(
    path_to_sihan_corpora_dict="/Users/cquan/Documents/model/stanford-segmenter-2018-10-16/data",
    path_to_model="/Users/cquan/Documents/model/stanford-segmenter-2018-10-16/data/pku.gz",
    path_to_dict="/Users/cquan/Documents/model/stanford-segmenter-2018-10-16/data/dict-chris6.ser.gz")
res = segmenter.segment(u'北海已成为中国对外开放中升起的一颗明星')
print(type(res))
print(res.encode('utf-8'))


from nltk.parse.stanford import StanfordParser
eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
res = list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split()))
for tree in res:
    print(tree)
    tree.draw()

ch_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz')
ch_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz')
res1 = list(ch_parser.parse(u'北海 已 成为 中国 对外开放 中 升起 的 一 颗 明星'.split()))
for tree in res1:
    print(tree)
    tree.draw()


from nltk.parse.stanford import StanfordDependencyParser
eng_parser = StanfordDependencyParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
res2 = list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split()))
# for row in res2[0].triples():
#     print(row)
for graph in res2:
    # print(graph)
    t = graph.tree()
    t.draw()
    # graph.draw()  # graph没有draw方法，不知道怎么绘图


from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/Users/cquan/Documents/model/stanford-corenlp-full-2018-10-05', lang='zh')
# 这里改成你stanford-corenlp所在的目录
# sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
sentence = '清华大学位于北京。'
print('Tokenize:', nlp.word_tokenize(sentence))
print('Part of Speech:', nlp.pos_tag(sentence))
print('Named Entities:', nlp.ner(sentence))
print('Constituency Parsing:', nlp.parse(sentence))
print('Dependency Parsing:', nlp.dependency_parse(sentence))

nlp.close()  # Do not forget to close! The backend server will consume a lot memery.
