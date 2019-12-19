import spacy
from spacy import displacy

# nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
# nlp = spacy.load('en_core_web_lg')  # 加载预训练模型
nlp = spacy.load('./zh_core_web_sm-2.0.5/zh_core_web_sm/zh_core_web_sm-2.0.5')  # 加载预训练模型


def tokens():
    text = 'A magnetic monopole is a hypothetical elementary particle.'
    doc = nlp(text)
    tokens = [token for token in doc]               # 将句子切分为单词
    tokens_pos = [token.pos_ for token in doc]      # 词性标注，Coarse-grained part-of-speech tag.
    tokens_tag = [token.tag_ for token in doc]      # 词性标注，Fine-grained part-of-speech tag.
    tokens_lemma = [token.lemma_ for token in doc]  # 词性还原
    tokens_stop = [token.is_stop for token in doc]  # 识别停止词
    tokens_dep = [token.dep_ for token in doc]      # 依存分析标签
    noun_chunks = [nc for nc in doc.noun_chunks]    # 提取名词短语
    print(tokens)
    print(tokens_pos)
    print(tokens_tag)
    print(tokens_lemma)
    print(tokens_stop)
    print(tokens_dep)
    print(noun_chunks)


def ner():
    txt = '''European authorities fined Google a record $5.1 billion
    on Wednesday for abusing its power in the mobile phone market and
    ordered the company to alter its practices
    '''
    # txt = '西门子将努力参与中国的三峡工程建设。'
    doc = nlp(txt)
    ners = [(ent.text, ent.label_) for ent in doc.ents]
    print(ners)
    displacy.serve(doc, style='ent')


def similarity():
    # 词汇语义相似度(关联性)
    banana = nlp.vocab['banana']
    banana_vec = nlp.vocab['banana'].vector  # 可以直接得到word2vec的vector，sm模型没有
    dog = nlp.vocab['dog']
    fruit = nlp.vocab['fruit']
    animal = nlp.vocab['animal']
    print(dog.similarity(animal), dog.similarity(fruit))  # 0.6618534 0.23552845
    print(banana.similarity(fruit), banana.similarity(animal))  # 0.67148364 0.2427285

    # 文本语义相似度(关联性)
    target = nlp("Cats are beautiful animals.")
    doc1 = nlp("Dogs are awesome.")
    doc2 = nlp("Some gorgeous creatures are felines.")
    doc3 = nlp("Dolphins are swimming mammals.")
    print(target.similarity(doc1))  # 0.8901765218466683
    print(target.similarity(doc2))  # 0.9115828449161616
    print(target.similarity(doc3))  # 0.7822956752876101


def coreference_resolution():
    txt = "My sister has a son and she loves him."

    # 将预训练的神经网络指代消解加入到spacy的管道中
    import neuralcoref
    neuralcoref.add_to_pipe(nlp)

    doc = nlp(txt)
    print(doc._.coref_clusters)


def display():
    txt = "In particle physics, a magnetic monopole is a hypothetical elementary particle."
    # txt = '西门子将努力参与中国的三峡工程建设。'
    displacy.serve(nlp(txt), style='dep', options={'distance': 90})

# tokens()
# ner()
# similarity()
# coreference_resolution()
display()