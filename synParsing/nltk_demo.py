import nltk
from nltk.corpus import conll2000, treebank_chunk  # 两个数据集


text1 = 'Lalbagh Botanical Garden is a well known botanical garden in Bengaluru, India.'
text2 = 'Ravi is the CEO of a company. He is very powerful public speaker also.'


def builtin_chunk(text):
    """
    内置分块器
    分块：从文本中抽取短语
    """
    # 文本切割成多个句子
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(words)
        chunks = nltk.ne_chunk(tags)
        print(chunks)


def diy_gram_chunk(text):
    """
    编写简单的RE分块器
    """
    # 词性语法规则
    grammar = '\n'.join([
        'NP: {<DT>*<NNP>}',  # 一个或多个DT后紧跟一个NNP
        'NP: {<JJ>*<NN>}',  # 一个或多个JJ后紧跟一个NN
        'NP: {<NNP>+}',  # 一个或多个NNP组成
    ])

    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(words)
        # 将语法规则放到RegexpParser对象中
        chunkparser = nltk.RegexpParser(grammar)
        result = chunkparser.parse(tags)
        print(result)


def train_parser():
    """
    训练分块器
    """
    # 简单的分块器，抽取NNP（专有名词）
    def mySimpleChunker():
        grammar = 'NP: {<NNP>+}'
        return nltk.RegexpParser(grammar)

    # 不抽取任何东西，只用于检验算法能否正常运行
    def test_nothing(data):
        cp = nltk.RegexpParser("")
        print(cp.evaluate(data))

    # 测试mySimpleChunker()函数
    def test_mySimpleChunker(data):
        schunker = mySimpleChunker()
        print(schunker.evaluate(data))

    datasets = [
        conll2000.chunked_sents('test.txt', chunk_types=['NP']),
        treebank_chunk.chunked_sents(),
    ]

    # 前50个IOB标注语句 计算分块器的准确率
    for dataset in datasets:
        test_nothing(dataset[:50])
        print('---------------------')
        test_mySimpleChunker(dataset[:50])
        print()


# 利用grammar创建CFG对象
grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> NNP VBZ
    VP -> IN NNP | DT NN IN NNP
    NNP -> 'Tajmahal' | 'Agra' | 'Bangalore' | 'Karnataka'
    VBZ -> 'is'
    IN -> 'in' | 'of'
    DT -> 'the'
    NN -> 'capital'
    """)

text3 = [
    "Tajmahal is in Agra",
    "Bangalore is the capital of Karnataka",
]


def RDParser():
    """
    递归下降句法分析
    递归先序遍历句法分析树
    NLTK的RD分析器
    """
    def RDParserExample(grammar, textlist):
        parser = nltk.parse.RecursiveDescentParser(grammar)
        for text in textlist:
            sentence = nltk.word_tokenize(text)
            for tree in parser.parse(sentence):
                print(tree)
                tree.draw()

    RDParserExample(grammar, text3)


def SRParser():
    """
    shift-reduce句法分析
    shift-reduce句法分析器：从左到右单线程，也可以从上到下多线程
    """
    def SRParserExample(grammar, textlist):
        parser = nltk.parse.ShiftReduceParser(grammar)
        for text in textlist:
            sentence = nltk.word_tokenize(text)
            for tree in parser.parse(sentence):
                print(tree)
                tree.draw()

    SRParserExample(grammar, text3)


def dependencyParser():
    """
    依存句法分析和主观依存分析
    """
    # 依存相关规则
    grammar = nltk.grammar.DependencyGrammar.fromstring("""
    'savings' -> 'small'
    'yield' -> 'savings'
    'gains' -> 'large'
    'yield' -> 'gains'
    """)

    sentence = 'small savings yield large gains'
    dp = nltk.parse.ProjectiveDependencyParser(grammar)
    print(sorted(dp.parse(sentence.split())))
    for t in sorted(dp.parse(sentence.split())):
        print(t)
        t.draw()


def chartParser():
    """
    线图句法分析
    """
    from nltk.grammar import CFG
    from nltk.parse.chart import ChartParser, BU_LC_STRATEGY

    # BNF格式文法 开始符号：S 终结符号：单词
    grammar = CFG.fromstring("""
    S -> T1 T4
    T1 -> NNP VBZ
    T2 -> DT NN
    T3 ->IN NNP
    T4 -> T3 | T2 T3
    NNP -> 'Tajmahal' | 'Agra' | 'Bangalore' | 'Karnataka'
    VBZ -> 'is'
    IN -> 'in' | 'of'
    DT -> 'the'
    NN -> 'capital'
    """)

    cp = ChartParser(grammar, BU_LC_STRATEGY, trace=True)
    # trace=True可以看见分析过程
    # strategy=BU_LC_STRATEGY是默认的，不写好像也行

    sentence = 'Bangalore is the capital of Karnataka'
    tokens = sentence.split()
    chart = cp.chart_parse(tokens)  # 对单词列表分析，并存到chart对象
    parses = list(chart.parses(grammar.start()))  # 将chart取到的所有分析树赋给parses
    print('Total Edges:', len(chart.edges()))  # 输出chart对象所有边的数量
    for tree in parses:
        print(tree)
        tree.draw()


if __name__ == '__main__':
    # dependencyParser()
    chartParser()
