# encoding:utf8
import gensim

if __name__ == '__main__':

    model = gensim.models.Word2Vec.load('model/baike_26g_news_13g_novel_229g.model')
    word1 = u'农业'
    word2 = u'书生'
    if word1 in model:
        print(u"'%s'的词向量为： " % word1)
        print(model[word1])
    else:
        print(u'单词不在字典中！')

    result = model.most_similar(word2)
    print(u"\n与'%s'最相似的词为： " % word2)
    for e in result:
        print('%s: %f' % (e[0], e[1]))

    print(u"\n'%s'与'%s'的相似度为： " % (word1, word2))
    print(model.similarity(word1, word2))

    print(u"\n'早餐 晚餐 午餐 中心'中的离群词为： ")
    print(model.doesnt_match(u"早餐 晚餐 午餐 中心".split()))

    print(u"\n与'%s'最相似，而与'%s'最不相似的词为： " % (word1, word2))
    temp = (model.most_similar(positive=[u'篮球'], negative=[u'计算机'], topn=1))
    print('%s: %s' % (temp[0][0], temp[0][1]))
