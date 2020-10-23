# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

vect = CountVectorizer()
vect.fit(corpus)
x = vect.transform(corpus)
# x = vect.fit_transform(corpus)  # 两步可以合成一步

feature_name = vect.get_feature_names()

print(feature_name)
print(x.toarray())
print(x)


# TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
v = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(v.fit_transform(corpus))
print(tfidf)


# TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
res = tfidf2.fit_transform(corpus)
print(res)
