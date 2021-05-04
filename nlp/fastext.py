import fasttext

# 训练词向量模型，Skipgram model :
# model = fasttext.train_unsupervised('enwik9', model='skipgram')
#
# print(model.words)
# print(model['king'])
#
# model.save_model("model_filename.bin")
# model = fasttext.load_model("model_filename.bin")
# print('king' in model)


# 训练分类模型，
model = fasttext.train_supervised('cooking.stackexchange.txt')
print(model.words)
print(model.labels)
print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Which baking dish is best to bake a banana bread ?", k=3))
print(model.predict(["Which baking dish is best to bake a banana bread ?", "Why not put knives in the dishwasher?"], k=3))

